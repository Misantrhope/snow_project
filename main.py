import os
import sys
import traceback
import time

if sys.platform == "win32":
    qt_dll_path = None
    try:
        import PyQt5

        pyqt_dir = os.path.dirname(PyQt5.__file__)
        for path in [os.path.join(pyqt_dir, "Qt5", "plugins", "platforms"),
                     os.path.join(pyqt_dir, "Qt", "plugins", "platforms")]:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "qwindows.dll")):
                qt_dll_path = path
                break
    except ImportError:
        try:
            import PySide6

            pyside_dir = os.path.dirname(PySide6.__file__)
            for path in [os.path.join(pyside_dir, "plugins", "platforms"),
                         os.path.join(pyside_dir, "Qt", "plugins", "platforms")]:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "qwindows.dll")):
                    qt_dll_path = path
                    break
        except ImportError:
            pass

    if qt_dll_path:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dll_path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import Normalize
from PIL import Image

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QFormLayout, QGroupBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
        QProgressBar, QTextEdit, QMenu, QAction, QMessageBox,
        QDialog, QDialogButtonBox, QScrollArea, QComboBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QUrl
    from PyQt5.QtGui import QFont, QDesktopServices

    QT_BINDING = "PyQt5"
    QT_HAS_EXEC_ = True
except ImportError:
    try:
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QFormLayout, QGroupBox, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
            QProgressBar, QTextEdit, QMenu, QAction, QMessageBox,
            QDialog, QDialogButtonBox, QScrollArea, QComboBox
        )
        from PySide6.QtCore import Qt, QThread, Signal, QUrl
        from PySide6.QtGui import QFont, QDesktopServices, QAction

        QT_BINDING = "PySide6"
        QT_HAS_EXEC_ = False
    except ImportError as e:
        raise SystemExit("Не установлен ни PyQt5, ни PySide6. Установите: pip install PyQt5") from e

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SNAPSHOTS_DIR = os.path.join(OUTPUT_DIR, "snapshots")
GIF_FRAMES_DIR = os.path.join(OUTPUT_DIR, "gif_frames")
GIF_PATH = os.path.join(OUTPUT_DIR, "snowflake_comparison.gif")
SPEED_GRAPH_PATH = os.path.join(OUTPUT_DIR, "speed_comparison.png")


class SnowflakeCA:
    def __init__(self, n=401, lattice="square", seed=0,
                 D=0.15, vapor_init=0.60, v_bound=1.0,
                 threshold=0.01, p_deposit=0.80, tip_bias=1.2,
                 mode="static"):
        self.n = n
        self.lattice = lattice
        self.D = D
        self.threshold = threshold
        self.v_bound = v_bound
        self.p_deposit = p_deposit
        self.tip_bias = tip_bias
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        if lattice == "square":
            self.offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif lattice == "hex":
            self.offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        else:
            raise ValueError("lattice must be 'square' or 'hex'")

        ci = n // 2
        self.ice = np.zeros((n, n), dtype=bool)
        self.ice[ci, ci] = True
        self.ice_age = np.zeros((n, n), dtype=float)

        self.vapor = np.full((n, n), vapor_init, dtype=float)
        self.vapor[self.ice] = 0.0
        self._fix_boundary(self.vapor)
        self.t = 0.0
        self.max_r = 0.0

    def _fix_boundary(self, a):
        a[0, :] = self.v_bound
        a[-1, :] = self.v_bound
        a[:, 0] = self.v_bound
        a[:, -1] = self.v_bound

    def _neighbor_sum(self, a):
        s = np.zeros_like(a, dtype=float)
        for di, dj in self.offsets:
            s += np.roll(a, shift=(di, dj), axis=(0, 1))
        return s

    def step(self):
        self.t += 1.0
        k = len(self.offsets)
        new_v = self.vapor + self.D * (self._neighbor_sum(self.vapor) / k - self.vapor)
        new_v[self.ice] = 0.0
        self._fix_boundary(new_v)
        self.vapor = np.clip(new_v, 0.0, self.v_bound)

        ice_neigh = self._neighbor_sum(self.ice.astype(float)) > 0
        eligible = (~self.ice) & ice_neigh
        eligible[0, :] = False
        eligible[-1, :] = False
        eligible[:, 0] = False
        eligible[:, -1] = False

        if self.mode == "static":
            newly = eligible & (self.vapor > self.threshold)
        else:
            span = self.v_bound - self.threshold + 1e-9
            excess = np.clip((self.vapor - self.threshold) / span, 0.0, 1.0)
            rate = self.p_deposit * excess ** self.tip_bias
            rnd = self.rng.random(self.vapor.shape)
            newly = eligible & (rnd < rate)

        if newly.any():
            self.ice |= newly
            self.ice_age[newly] = self.t
            self.vapor[self.ice] = 0.0
            ci = self.n // 2
            ys, xs = np.nonzero(newly)
            d = float(np.hypot(ys - ci, xs - ci).max())
            if d > self.max_r:
                self.max_r = d

    def polar_profile(self, nbins=120):
        ys, xs = np.nonzero(self.ice)
        thetas = np.linspace(0, 2 * np.pi, nbins, endpoint=False)
        if len(ys) < 3:
            return np.zeros(nbins), thetas
        ci = self.n // 2
        dy, dx = ys - ci, xs - ci
        r = np.hypot(dy, dx)
        th = np.arctan2(dy, dx) % (2 * np.pi)
        bins = np.linspace(0, 2 * np.pi, nbins + 1)
        idx = np.clip(np.digitize(th, bins) - 1, 0, nbins - 1)
        rmax = np.zeros(nbins)
        np.maximum.at(rmax, idx, r)
        return rmax, 0.5 * (bins[:-1] + bins[1:])


def snapshot(sim):
    return dict(vapor=sim.vapor.copy(),
                ice=sim.ice.copy(),
                ice_age=sim.ice_age.copy(),
                t=sim.t, r=sim.max_r)


def save_snapshot(vapor, ice, ice_age, t, name, step, half,
                  t_max, n, save_path):
    fig = Figure(figsize=(6, 6), dpi=130, constrained_layout=True)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    cx = cy = n // 2
    ax.imshow(vapor, cmap="Blues", vmin=0, vmax=1.0,
              origin="lower", interpolation="nearest", alpha=0.55)
    age = np.where(ice, t - np.where(ice, ice_age, t), np.nan)
    ax.imshow(age, cmap="magma", vmin=0, vmax=t_max,
              origin="lower", interpolation="nearest")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}\nшаг {step}", fontsize=11)
    fig.savefig(save_path, dpi=130)


def assemble_gif(frame_paths, gif_path, fps, max_dim=500):
    imgs = []
    target_size = None
    for p in frame_paths:
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        if target_size is None:
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                target_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            else:
                target_size = img.size
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        imgs.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
    if not imgs:
        return False
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return True


def plot_speed_comparison(times_sq, times_hex, save_path):
    fig = Figure(figsize=(8, 5), dpi=100, constrained_layout=True)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.plot(times_sq, label="Квадратная (4 соседа)", color="#1f77b4", alpha=0.8)
    ax.plot(times_hex, label="Гексагональная (6 соседей)", color="#d62728", alpha=0.8)

    ax.set_xlabel("Шаг симуляции")
    ax.set_ylabel("Время выполнения шага, сек")
    ax.set_title("Сравнение вычислительной скорости решёток")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=100)


class SnowflakeAnimator:
    def __init__(self, N=401, seed=7, n_steps=500, interp=6,
                 init_half=12.0, margin=8.0, save_every=100,
                 common=None, fig_dpi=100):
        self.N = N
        self.seed = seed
        self.n_steps = n_steps
        self.interp = interp
        self.n_frames = n_steps * interp
        self.init_half = init_half
        self.margin = margin
        self.save_every = save_every
        self.common = common or {}

        self.names = ["square_lattice", "hex_lattice"]
        self.names_ru = ["Прямоугольная, 4 соседа", "Гексагональная, 6 соседей"]
        self.save_steps = set(range(0, n_steps + 1, save_every)) | {n_steps}

        self.time_sq = []
        self.time_hex = []

        self._reset_state()
        self._setup_figure(fig_dpi)

    def _reset_state(self):
        self.sims = [
            SnowflakeCA(n=self.N, lattice="square", seed=self.seed, **self.common),
            SnowflakeCA(n=self.N, lattice="hex", seed=self.seed, **self.common),
        ]
        self.state_prev = [snapshot(self.sims[0]), snapshot(self.sims[1])]
        for sim in self.sims:
            sim.step()
        self.state_curr = [snapshot(self.sims[0]), snapshot(self.sims[1])]
        self.saved_snapshots = set()

    def _setup_figure(self, fig_dpi):
        self.fig = Figure(figsize=(11, 8), dpi=fig_dpi, constrained_layout=True)
        FigureCanvas(self.fig)
        self.fig.patch.set_facecolor("white")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0])
        self.ax_sq = self.fig.add_subplot(gs[0, 0])
        self.ax_hex = self.fig.add_subplot(gs[0, 1])
        self.ax_polar = self.fig.add_subplot(gs[1, :], projection="polar")

        self.t_max = float(self.n_steps) + 1.0
        self.age_norm = Normalize(vmin=0, vmax=self.t_max)
        try:
            self.age_cmap = matplotlib.colormaps["magma"]
        except AttributeError:
            import matplotlib.cm as cm
            self.age_cmap = cm.get_cmap("magma")

        self.vapor_imgs, self.ice_imgs, self.titles = [], [], []
        for ax, sim, name in zip([self.ax_sq, self.ax_hex],
                                 self.sims, self.names_ru):
            bg = ax.imshow(sim.vapor, cmap="Blues", vmin=0, vmax=sim.v_bound,
                           origin="lower", interpolation="bilinear", alpha=0.55)
            rgba = np.zeros((self.N, self.N, 4))
            cr = ax.imshow(rgba, origin="lower", interpolation="bilinear")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            self.vapor_imgs.append(bg)
            self.ice_imgs.append(cr)
            self.titles.append(ax.set_title(name, fontsize=11))

        self.line_sq, = self.ax_polar.plot([], [], lw=2.2,
                                           label=self.names_ru[0], color="#1f77b4")
        self.line_hex, = self.ax_polar.plot([], [], lw=2.2,
                                            label=self.names_ru[1], color="#d62728")
        self.ax_polar.set_title("Профиль r_max(θ): 4 vs 6 лепестков", fontsize=10)
        self.ax_polar.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5), fontsize=9)
        self.ax_polar.set_theta_zero_location("E")

        self.cx = self.cy = self.N // 2
        self.fig.suptitle("Рост снежинки из пара: сравнение решёток", fontsize=11)

    def save_step0_snapshots(self):
        for idx, name in enumerate(self.names):
            s = self.state_prev[idx]
            save_snapshot(s["vapor"], s["ice"], s["ice_age"], s["t"],
                          name, 0, self.init_half, self.n_steps, self.N,
                          os.path.join(SNAPSHOTS_DIR, f"{name}_step_0000.png"))
        self.saved_snapshots.add(0)

    def _save_current_snapshot(self, step):
        half_now = max(self.init_half,
                       max(self.state_curr[0]["r"], self.state_curr[1]["r"]) + self.margin)
        for idx, name in enumerate(self.names):
            c = self.state_curr[idx]
            save_snapshot(c["vapor"], c["ice"], c["ice_age"], c["t"],
                          name, step, half_now, self.n_steps, self.N,
                          os.path.join(SNAPSHOTS_DIR, f"{name}_step_{step:04d}.png"))
        self.saved_snapshots.add(step)

    def render(self, frame):
        s = (frame % self.interp) / self.interp
        for idx in range(2):
            p = self.state_prev[idx]
            c = self.state_curr[idx]
            v_disp = (1.0 - s) * p["vapor"] + s * c["vapor"]
            w_disp = (1.0 - s) * p["ice"].astype(float) + s * c["ice"].astype(float)
            age_p = np.where(p["ice"], p["t"] - p["ice_age"], 0.0)
            age_c = np.where(c["ice"], c["t"] - c["ice_age"], 0.0)
            age_disp = (1.0 - s) * age_p + s * age_c
            rgba = self.age_cmap(self.age_norm(np.clip(age_disp, 0.0, self.t_max)))
            rgba[..., 3] = np.clip(w_disp, 0.0, 1.0) * 0.95
            self.vapor_imgs[idx].set_data(v_disp)
            self.ice_imgs[idx].set_data(rgba)
            self.titles[idx].set_text(
                f"{self.names_ru[idx]}\nшаг {int(round(c['t']))}, "
                f"клеток снежинки: {int(c['ice'].sum())}")

        r_p = max(self.state_prev[0]["r"], self.state_prev[1]["r"])
        r_c = max(self.state_curr[0]["r"], self.state_curr[1]["r"])
        r_now = (1.0 - s) * r_p + s * r_c
        half = max(self.init_half, r_now + self.margin)
        for ax in (self.ax_sq, self.ax_hex):
            ax.set_xlim(self.cx - half, self.cx + half)
            ax.set_ylim(self.cy - half, self.cy + half)

        r_sq, th = self.sims[0].polar_profile()
        r_hex, _ = self.sims[1].polar_profile()
        self.line_sq.set_data(np.append(th, th[0]), np.append(r_sq, r_sq[0]))
        self.line_hex.set_data(np.append(th, th[0]), np.append(r_hex, r_hex[0]))
        self.ax_polar.set_ylim(0, max(self.init_half, half) * 1.1)

        if (frame + 1) % self.interp == 0 and (frame + 1) < self.n_frames:
            self.state_prev = self.state_curr

            if frame % 2 == 0:
                t0 = time.perf_counter()
                self.sims[0].step()
                t1 = time.perf_counter()
                self.sims[1].step()
                t2 = time.perf_counter()
                self.time_sq.append(t1 - t0)
                self.time_hex.append(t2 - t1)
            else:
                t0 = time.perf_counter()
                self.sims[1].step()
                t1 = time.perf_counter()
                self.sims[0].step()
                t2 = time.perf_counter()
                self.time_hex.append(t1 - t0)
                self.time_sq.append(t2 - t1)

            self.state_curr = [snapshot(self.sims[0]), snapshot(self.sims[1])]
            cur_step = int(round(self.sims[0].t))
            if cur_step in self.save_steps and cur_step not in self.saved_snapshots:
                self._save_current_snapshot(cur_step)


PRESETS = {
    "potato": dict(N=201, n_steps=200, interp=2, fps=10, dpi=40,
                   gif_skip=2, save_every=100),
    "medium": dict(N=401, n_steps=400, interp=4, fps=10, dpi=60,
                   gif_skip=4, save_every=100),
    "excellent": dict(N=601, n_steps=800, interp=8, fps=15, dpi=100,
                      gif_skip=10, save_every=100),
}

DEFAULT_PHYSICS = dict(
    seed=7, D=0.15, vapor_init=0.60, v_bound=1.0,
    threshold=0.01, p_deposit=0.80, tip_bias=1.2,
    mode="static"
)

HTML_HELP_PARAMS = """
<html><body style="font-family: sans-serif; font-size: 11pt; line-height: 1.5;">
<h2>Инструкция по работе</h2>
<ol>
<li><b>Выберите пресет качества</b> в левом верхнем углу. Это автоматически
настроит размер поля, разрешение и количество шагов.</li>
<li><b>Настройте физические параметры</b> (опционально). Значения по умолчанию
подобраны так, чтобы получалась симметричная снежинка.</li>
<li>Нажмите <b>«Запустить симуляцию»</b>. Программа рассчитает модель и
сохранит результат.</li>
<li>После завершения нажмите <b>«Открыть GIF»</b>, чтобы посмотреть
анимацию, или <b>«Открыть папку»</b>, чтобы найти промежуточные снимки.</li>
</ol>

<h2>Описание параметров</h2>
<h3>Физика</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
<tr><th align="left" width="28%">Параметр</th><th align="left">Смысл</th></tr>
<tr><td><b>Режим</b></td>
<td><b>Статика</b> (детерминированный) — клетка замерзает строго при
превышении порога пара. Даёт идеальные фракталы без случайного шума.<br>
<b>Хаос</b> (вероятностный) — замерзание зависит от случайного числа.
Появляются случайные боковые веточки.</td></tr>
<tr><td><b>D</b></td>
<td>Коэффициент диффузии пара. Больше D — пар быстрее подтекает к фронту,
снежинка растёт быстрее и «пышнее».</td></tr>
<tr><td><b>vapor_init</b></td>
<td>Стартовая концентрация пара в каждой клетке. Если мало — снежинка
быстро «выест» пар вокруг и остановится.</td></tr>
<tr><td><b>v_bound</b></td>
<td>Концентрация пара на границе. Резервуар, подпитывающий область.</td></tr>
<tr><td><b>threshold</b></td>
<td>Порог замерзания. При локальном паре ниже threshold клетка не
замерзает.</td></tr>
<tr><td><b>p_deposit</b></td>
<td>Вероятность замерзания. Влияет только в режиме «Хаос».
Больше — фронт агрессивнее, форма грубее.</td></tr>
<tr><td><b>tip_bias</b></td>
<td>Нелинейность скорости замерзания. Влияет только в режиме «Хаос».
Больше tip_bias — сильнее выражены ветви.</td></tr>
<tr><td><b>seed</b></td>
<td>Зерно генератора случайных чисел. Одинаковый seed → одинаковая
снежинка при прочих равных.</td></tr>
</table>

<h3>Симуляция</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
<tr><th align="left" width="28%">Параметр</th><th align="left">Смысл</th></tr>
<tr><td><b>N</b></td>
<td>Размер сетки (N × N клеток). Больше N — крупнее снежинка,
но медленнее расчёт.</td></tr>
<tr><td><b>Шагов</b></td>
<td>Число шагов клеточного автомата. 300 — быстрый тест, 500 — норм,
800 — подробный рост.</td></tr>
</table>

<h3>Вывод</h3>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">
<tr><th align="left" width="28%">Параметр</th><th align="left">Смысл</th></tr>
<tr><td><b>interp</b></td>
<td>Кадров анимации на 1 CA-шаг. Больше — плавнее, но больше кадров
рендерить.</td></tr>
<tr><td><b>fps</b></td>
<td>Кадров в секунду в GIF.</td></tr>
<tr><td><b>dpi</b></td>
<td>Разрешение кадра. 50 — быстро и грубо, 100 — чётко, но тяжелее.</td></tr>
<tr><td><b>gif_skip</b></td>
<td>Берём каждый N-й отрендеренный кадр в GIF. Больше — файл легче.</td></tr>
<tr><td><b>save_every</b></td>
<td>Шаг между PNG-снимками в output/snapshots/.</td></tr>
</table>
</body></html>
"""

HTML_HELP_GRAPH = """
<html><body style="font-family: sans-serif; font-size: 11pt; line-height: 1.5;">
<h2>Что изображено на графике (GIF)</h2>
<p>Итоговый GIF-файл состоит из трех областей: двух панелей с полем и одной
панели с полярным графиком.</p>

<h3>Верхние панели (Симуляция)</h3>
<ul>
<li><b>Слева вверху:</b> рост снежинки на квадратной решётке (у каждой
клетки 4 соседа).</li>
<li><b>Справа вверху:</b> рост снежинки на гексагональной решётке
(у каждой клетки 6 соседей).</li>
</ul>

<h3>Цветовая схема</h3>
<p>На верхних панелях наложены два слоя данных:</p>
<ul>
<li><b>Фон (синяя палитра Blues):</b> концентрация водяного пара.
Чем темнее и насыщеннее синий цвет, тем больше пара в данной точке.
Белый цвет — пар отсутствует.</li>
<li><b>Снежинка (палитра magma):</b> замерзшие клетки. Цвет показывает
<b>возраст</b> льда. Ярко-желтые клетки — это только что замерзший фронт.
Темно-фиолетовые и черные клетки — это основание снежинки, которая замерзла
давно.</li>
</ul>

<h3>Нижняя панель (Полярный профиль)</h3>
<p>Здесь показано, как далеко (на какое расстояние от центра) успела вырасти
снежинка в каждом направлении (угле от 0 до 360 градусов).</p>
<ul>
<li><b>Синяя линия:</b> профиль для квадратной решётки. Вы увидите
<b>4 лепестка</b>. Это означает, что на квадратной сетке снежинка растет
в основном вверх, вниз, влево и вправо.</li>
<li><b>Красная линия:</b> профиль для гексагональной решётки. Вы увидите
<b>6 лепестков</b>. Это означает, что сетка навязывает снежинке
правильную шестилучевую симметрию, как у настоящего льда.</li>
</ul>
</body></html>
"""


class SimWorker(QThread):
    progress = Signal(int, int, str)
    log = Signal(str)
    done_ok = Signal(str, str, str)
    done_err = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = dict(params)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            gif_path, snap_dir, speed_msg = self._run_sim()
            if not self._abort:
                self.done_ok.emit(gif_path, snap_dir, speed_msg)
        except Exception as e:
            self.done_err.emit(f"{e}\n\n{traceback.format_exc()}")

    def _run_sim(self):
        p = self.params
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        os.makedirs(GIF_FRAMES_DIR, exist_ok=True)

        for f in os.listdir(GIF_FRAMES_DIR):
            if f.endswith(".png"):
                try:
                    os.remove(os.path.join(GIF_FRAMES_DIR, f))
                except OSError:
                    pass

        total_frames = p["n_steps"] * p["interp"]
        gif_frames = total_frames // max(1, p["gif_skip"])
        self.log.emit(f"Параметры: N={p['N']}, шагов={p['n_steps']}, "
                      f"interp={p['interp']}, dpi={p['dpi']}, режим={p['mode']}")
        self.log.emit(f"Кадров к рендеру: {total_frames}, "
                      f"из них в GIF: ~{gif_frames}")

        anim = SnowflakeAnimator(
            N=p["N"], seed=p["seed"], n_steps=p["n_steps"], interp=p["interp"],
            init_half=12.0, margin=8.0, save_every=p["save_every"],
            common=dict(D=p["D"], vapor_init=p["vapor_init"],
                        v_bound=p["v_bound"], threshold=p["threshold"],
                        p_deposit=p["p_deposit"], tip_bias=p["tip_bias"],
                        mode=p["mode"]),
            fig_dpi=100,
        )
        anim.fig.set_dpi(p["dpi"])

        self.log.emit("Сохранение шага 0...")
        anim.save_step0_snapshots()

        frame_paths = []

        anim.render(0)
        p0 = os.path.join(GIF_FRAMES_DIR, "frame_00000.png")
        anim.fig.savefig(p0)
        frame_paths.append(p0)

        for frame in range(1, anim.n_frames):
            if self._abort:
                self.log.emit("Прервано пользователем")
                return "", "", ""

            anim.render(frame)

            if frame % p["gif_skip"] == 0:
                fp = os.path.join(GIF_FRAMES_DIR, f"frame_{frame:05d}.png")
                anim.fig.savefig(fp)
                frame_paths.append(fp)

            if frame % 5 == 0:
                self.progress.emit(frame, anim.n_frames,
                                   f"кадр {frame}/{anim.n_frames}, "
                                   f"CA-шаг {int(anim.sims[0].t)}/{p['n_steps']}")

        self.progress.emit(anim.n_frames, anim.n_frames, "Сборка GIF...")
        self.log.emit(f"Собрано {len(frame_paths)} PNG, упаковка в GIF...")

        ok = assemble_gif(frame_paths, GIF_PATH, p["fps"], max_dim=500)
        if not ok:
            raise RuntimeError("Не удалось собрать GIF (нет кадров)")

        self.log.emit("Построение графика скорости...")
        plot_speed_comparison(anim.time_sq, anim.time_hex, SPEED_GRAPH_PATH)

        avg_sq = np.mean(anim.time_sq) if anim.time_sq else 0
        avg_hex = np.mean(anim.time_hex) if anim.time_hex else 0
        diff = ((avg_hex - avg_sq) / avg_sq * 100) if avg_sq > 0 else 0

        speed_msg = (f"Среднее время шага: Квадратная {avg_sq * 1000:.3f} мс, "
                     f"Гексагональная {avg_hex * 1000:.3f} мс.")
        self.log.emit(speed_msg)

        print(f"\n[РЕЗУЛЬТАТ] Гексагональная решетка медленнее квадратной на {diff:.1f}%")

        size_mb = os.path.getsize(GIF_PATH) / (1024 * 1024)
        self.log.emit(f"GIF сохранен: {GIF_PATH} ({size_mb:.1f} MB)")
        self.log.emit(f"Снимки: {sorted(anim.saved_snapshots)}")

        return GIF_PATH, SNAPSHOTS_DIR, speed_msg


class HelpDialog(QDialog):
    def __init__(self, parent, title, html):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 650)
        layout = QVBoxLayout(self)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(html)
        layout.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Рост снежинки: клеточный автомат")
        self.resize(980, 800)

        self.worker = None
        self.last_gif = None
        self.last_snapshots = None
        self.last_speed_msg = ""

        self._build_menu()
        self._build_ui()
        self._apply_preset("medium")
        self._set_default_physics()
        self.log.append("Готов к работе. Стартовый пресет: Medium, режим: Статика.")

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("Файл")
        m_file.addAction(QAction("Открыть папку output", self,
                                 triggered=self._open_output))
        m_file.addSeparator()
        m_file.addAction(QAction("Выход", self, triggered=self.close))

        m_help = mb.addMenu("Справка")
        m_help.addAction(QAction("Инструкция и параметры", self, triggered=self._show_help_params))
        m_help.addAction(QAction("Описание графиков", self, triggered=self._show_help_graph))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        left = QVBoxLayout()

        gb_preset = QGroupBox("Пресеты качества")
        ph = QHBoxLayout(gb_preset)
        for label, key in [("Potato", "potato"),
                           ("Medium", "medium"),
                           ("Excellent", "excellent")]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, k=key: self._apply_preset(k))
            ph.addWidget(b)
        left.addWidget(gb_preset)

        gb_phys = QGroupBox("Физика и режим")
        f_phys = QFormLayout(gb_phys)
        self.sp_seed = QSpinBox();
        self.sp_seed.setRange(0, 999999)
        self.sp_D = QDoubleSpinBox();
        self.sp_D.setRange(0.01, 1.0);
        self.sp_D.setSingleStep(0.01);
        self.sp_D.setDecimals(3)
        self.sp_vinit = QDoubleSpinBox();
        self.sp_vinit.setRange(0.1, 1.0);
        self.sp_vinit.setSingleStep(0.05);
        self.sp_vinit.setDecimals(2)
        self.sp_vbound = QDoubleSpinBox();
        self.sp_vbound.setRange(0.5, 2.0);
        self.sp_vbound.setSingleStep(0.1);
        self.sp_vbound.setDecimals(2)
        self.sp_thresh = QDoubleSpinBox();
        self.sp_thresh.setRange(0.01, 0.5);
        self.sp_thresh.setSingleStep(0.01);
        self.sp_thresh.setDecimals(3)
        self.sp_pdep = QDoubleSpinBox();
        self.sp_pdep.setRange(0.05, 1.0);
        self.sp_pdep.setSingleStep(0.05);
        self.sp_pdep.setDecimals(2)
        self.sp_tip = QDoubleSpinBox();
        self.sp_tip.setRange(1.0, 4.0);
        self.sp_tip.setSingleStep(0.1);
        self.sp_tip.setDecimals(2)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Статика (Детерминированный)", "Хаос (Вероятностный)"])

        f_phys.addRow("Режим:", self.cb_mode)
        f_phys.addRow("seed:", self.sp_seed)
        f_phys.addRow("D (диффузия):", self.sp_D)
        f_phys.addRow("vapor_init:", self.sp_vinit)
        f_phys.addRow("v_bound:", self.sp_vbound)
        f_phys.addRow("threshold:", self.sp_thresh)
        f_phys.addRow("p_deposit:", self.sp_pdep)
        f_phys.addRow("tip_bias:", self.sp_tip)
        left.addWidget(gb_phys)

        gb_sim = QGroupBox("Симуляция")
        f_sim = QFormLayout(gb_sim)
        self.sp_N = QSpinBox();
        self.sp_N.setRange(100, 1000);
        self.sp_N.setSingleStep(50)
        self.sp_steps = QSpinBox();
        self.sp_steps.setRange(50, 2000);
        self.sp_steps.setSingleStep(50)
        f_sim.addRow("N (размер сетки):", self.sp_N)
        f_sim.addRow("Шагов:", self.sp_steps)
        left.addWidget(gb_sim)

        gb_out = QGroupBox("Вывод")
        f_out = QFormLayout(gb_out)
        self.sp_interp = QSpinBox();
        self.sp_interp.setRange(1, 15)
        self.sp_fps = QSpinBox();
        self.sp_fps.setRange(5, 30)
        self.sp_dpi = QSpinBox();
        self.sp_dpi.setRange(40, 150);
        self.sp_dpi.setSingleStep(10)
        self.sp_gifskip = QSpinBox();
        self.sp_gifskip.setRange(1, 20)
        self.sp_saveevery = QSpinBox();
        self.sp_saveevery.setRange(50, 500);
        self.sp_saveevery.setSingleStep(50)
        f_out.addRow("interp:", self.sp_interp)
        f_out.addRow("fps:", self.sp_fps)
        f_out.addRow("dpi:", self.sp_dpi)
        f_out.addRow("gif_skip:", self.sp_gifskip)
        f_out.addRow("save_every:", self.sp_saveevery)
        left.addWidget(gb_out)

        btn_reset = QPushButton("Сбросить физику к значениям по умолчанию")
        btn_reset.clicked.connect(self._set_default_physics)
        left.addWidget(btn_reset)
        left.addStretch()

        left_w = QWidget();
        left_w.setLayout(left)
        scroll = QScrollArea();
        scroll.setWidget(left_w)
        scroll.setWidgetResizable(True);
        scroll.setMaximumWidth(400)
        main.addWidget(scroll)

        right = QVBoxLayout()

        gb_ctrl = QGroupBox("Управление")
        ctrl = QVBoxLayout(gb_ctrl)
        self.btn_start = QPushButton("Запустить симуляцию")
        self.btn_start.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        self.btn_start.clicked.connect(self._start)
        ctrl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Прервать")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_stop)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        ctrl.addWidget(self.progress)

        self.lbl_status = QLabel("Готов к запуску")
        ctrl.addWidget(self.lbl_status)
        right.addWidget(gb_ctrl)

        gb_log = QGroupBox("Лог")
        ll = QVBoxLayout(gb_log)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        ll.addWidget(self.log)
        right.addWidget(gb_log, 1)

        gb_res = QGroupBox("Результат")
        rl = QVBoxLayout(gb_res)
        self.lbl_result = QLabel("Симуляция ещё не запускалась")
        self.lbl_result.setWordWrap(True)
        rl.addWidget(self.lbl_result)

        row = QHBoxLayout()
        btn_gif = QPushButton("Открыть GIF")
        btn_gif.clicked.connect(self._open_gif)
        row.addWidget(btn_gif)
        btn_dir = QPushButton("Открыть папку")
        btn_dir.clicked.connect(self._open_output)
        row.addWidget(btn_dir)

        btn_speed = QPushButton("Открыть график скорости")
        btn_speed.clicked.connect(self._open_speed_graph)
        row.addWidget(btn_speed)
        rl.addLayout(row)
        right.addWidget(gb_res)

        right_w = QWidget();
        right_w.setLayout(right)
        main.addWidget(right_w, 1)

    def _set_default_physics(self):
        d = DEFAULT_PHYSICS
        self.sp_seed.setValue(d["seed"])
        self.sp_D.setValue(d["D"])
        self.sp_vinit.setValue(d["vapor_init"])
        self.sp_vbound.setValue(d["v_bound"])
        self.sp_thresh.setValue(d["threshold"])
        self.sp_pdep.setValue(d["p_deposit"])
        self.sp_tip.setValue(d["tip_bias"])
        if d["mode"] == "static":
            self.cb_mode.setCurrentIndex(0)
        else:
            self.cb_mode.setCurrentIndex(1)

    def _apply_preset(self, key):
        p = PRESETS[key]
        self.sp_N.setValue(p["N"])
        self.sp_steps.setValue(p["n_steps"])
        self.sp_interp.setValue(p["interp"])
        self.sp_fps.setValue(p["fps"])
        self.sp_dpi.setValue(p["dpi"])
        self.sp_gifskip.setValue(p["gif_skip"])
        self.sp_saveevery.setValue(p["save_every"])
        self.log.append(f"[preset] применён: {key}")

    def _collect(self):
        mode_str = "static" if self.cb_mode.currentIndex() == 0 else "chaos"
        return dict(
            N=self.sp_N.value(), seed=self.sp_seed.value(),
            n_steps=self.sp_steps.value(),
            D=self.sp_D.value(), vapor_init=self.sp_vinit.value(),
            v_bound=self.sp_vbound.value(), threshold=self.sp_thresh.value(),
            p_deposit=self.sp_pdep.value(), tip_bias=self.sp_tip.value(),
            interp=self.sp_interp.value(), fps=self.sp_fps.value(),
            dpi=self.sp_dpi.value(), gif_skip=self.sp_gifskip.value(),
            save_every=self.sp_saveevery.value(), mode=mode_str
        )

    def _start(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Уже работает", "Симуляция уже идёт.")
            return

        p = self._collect()
        if p["N"] < 100:
            QMessageBox.warning(self, "Ошибка", "N должно быть >= 100")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.lbl_status.setText("Запуск...")
        self.lbl_result.setText("Симуляция выполняется...")
        self.log.append("-" * 50)
        self.log.append("Новый запуск")

        self.worker = SimWorker(p)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log.append)
        self.worker.done_ok.connect(self._on_done)
        self.worker.done_err.connect(self._on_error)
        self.worker.start()

    def _stop(self):
        if self.worker is not None:
            self.worker.abort()
            self.lbl_status.setText("Прерывание...")
            self.btn_stop.setEnabled(False)

    def _on_progress(self, cur, total, msg):
        pct = int(100 * cur / max(1, total))
        self.progress.setValue(pct)
        self.lbl_status.setText(f"{msg} ({pct}%)")

    def _on_done(self, gif_path, snap_dir, speed_msg):
        self.last_gif = gif_path
        self.last_snapshots = snap_dir
        self.last_speed_msg = speed_msg
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(100)
        self.lbl_status.setText("Готово")
        self.lbl_result.setText(
            f"<b>Успешно завершено.</b><br><br>"
            f"{speed_msg}<br><br>"
            f"GIF: <code>output/snowflake_comparison.gif</code><br>"
            f"Снимки: <code>output/snapshots/</code><br>"
            f"График скорости: <code>output/speed_comparison.png</code>"
        )

    def _on_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_status.setText("Ошибка")
        self.log.append(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Ошибка симуляции", msg[:600])

    def _open_gif(self):
        if self.last_gif and os.path.exists(self.last_gif):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.last_gif)))
        else:
            QMessageBox.information(self, "Нет файла", "Сначала запустите симуляцию.")

    def _open_speed_graph(self):
        if os.path.exists(SPEED_GRAPH_PATH):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(SPEED_GRAPH_PATH)))
        else:
            QMessageBox.information(self, "Нет файла", "График скорости еще не построен.")

    def _open_output(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(OUTPUT_DIR))

    def _show_help_params(self):
        HelpDialog(self, "Инструкция и параметры", HTML_HELP_PARAMS).exec_()

    def _show_help_graph(self):
        HelpDialog(self, "Описание графиков", HTML_HELP_GRAPH).exec_()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            ret = QMessageBox.question(self, "Выход",
                                       "Симуляция выполняется. Прервать и выйти?",
                                       QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes:
                self.worker.abort()
                self.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    except OSError:
        pass

    app = QApplication.instance()
    if app is None:
        argv = sys.argv if sys.argv else ["snowflake"]
        app = QApplication(argv)

    app.setStyle("Fusion")

    win = MainWindow()
    win.show()

    try:
        if QT_HAS_EXEC_:
            sys.exit(app.exec_())
        else:
            sys.exit(app.exec())
    except SystemExit:
        raise
    except Exception as e:
        print(f"app.exec() упал: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()