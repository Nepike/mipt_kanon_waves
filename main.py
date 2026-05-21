
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

D     = 1.0    # коэффициент скорости: c^2(x) = D*x
PY0   = 0.5    # фиксированная компонента импульса  p_y⁰
H     = 1.0    # постоянная Планка
N_MAX = 5      # число уровней E_n
EPS   = 1e-4   # отступ от x=0 для численных расчётов


# ЛАГРАНЖЕВО МНОГООБРАЗИЕ  L^2
def px_upper(x, E):
    """
    Верхняя ветвь многообразия: p_x = +sqrt(E^2/(Dx) − py0^2).
    Действительна при x ≤ x0 = E^2/(D*py0^2).
    """
    return np.sqrt(np.maximum(E**2 / (D * x) - PY0**2, 0.0))


def turning_point(E):
    """Точка поворота (вершина параболы): x0 = E^2/(D*py0^2)."""
    return E**2 / (D * PY0**2)


def fig1_manifold(E_single=1.0, save='fig1_manifold.png'):
    """
    Показывает две ветви, направление обхода gamma_1 и индексы Маслова.
    """
    x   = np.linspace(EPS, 1.0, 3000)
    pxp =  px_upper(x, E_single)
    pxn = -px_upper(x, E_single)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, pxp, color='royalblue', lw=2.2,
            label=r'$p_x > 0$ (волна → стенка)')
    ax.plot(x, pxn, color='tomato',    lw=2.2,
            label=r'$p_x < 0$ (волна → берег)')

    # Границы
    ax.axvline(0, color='saddlebrown', lw=1.8, ls='--')
    ax.axvline(1, color='steelblue',   lw=1.8, ls='--')
    ax.axhline(0, color='gray',        lw=0.8, ls=':')

    # Стрелки — направление обхода
    mid = len(x) // 2
    for branch, color in [(pxp, 'royalblue'), (pxn, 'tomato')]:
        sign  = 1 if color == 'royalblue' else -1
        idx   = mid + sign * 40
        ax.annotate('', xy=(x[idx + sign*30], branch[idx + sign*30]),
                    xytext=(x[idx], branch[idx]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0))

    # Разрыв на стенке x=1
    px_wall = px_upper(np.array([1.0]), E_single)[0]
    ax.annotate('', xy=(1.0, -px_wall), xytext=(1.0, px_wall),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.0,
                                connectionstyle='arc3,rad=0.45'))

    # Замыкание на берегу
    ax.annotate('', xy=(0.04, pxn[30]), xytext=(0.04, pxp[30]),
                arrowprops=dict(arrowstyle='->', color='saddlebrown', lw=2.0,
                                connectionstyle='arc3,rad=-0.5'))

    # Подписи
    ax.text(0.02, 0, 'берег $x=0$\n$\\Delta m=1$',
            color='saddlebrown', fontsize=10, va='center')
    ax.text(1.03, 0, 'стенка $x=1$\n$\\Delta m=0$',
            color='steelblue',   fontsize=10, va='center')
    ax.text(0.48, 0.25, r'$\gamma_1$', fontsize=15, fontstyle='italic')

    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$p_x$', fontsize=13)
    ax.set_title(
        r'Проекция $L^2$ на плоскость $(x,\,p_x)$''\n'
        r'$p_x = \pm\sqrt{E^2/(Dx) - (p_y^0)^2}$,  '
        f'$D={D},\\ E={E_single},\\ p_y^0={PY0}$', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.07, 1.25)
    ax.set_ylim(-4.0, 4.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f'Сохранён: {save}')
    return fig


# ШАГ 2.  УСЛОВИЕ КВАНТОВАНИЯ И УРОВНИ E_n
def I_analytical(E):
    A = E / np.sqrt(D)
    B = PY0
    return (A**2 / B) * np.arcsin(B / A) + np.sqrt(A**2 - B**2)


def I_numerical(E):
    """Численная проверка I(E) через прямое интегрирование."""
    return quad(lambda x: np.sqrt(max(E**2 / (D * x) - PY0**2, 0.0)),
                1e-10, 1.0)[0]


def quant_equation(E, n):
    """
    Условие квантования БЗМ (mu=1):
      2*I(E) − pi*h/2*mu = 2pi*h*n  =>  I(E_n) = pi*h*(n + 1/4)
    """
    return I_analytical(E) - np.pi * H * (n + 0.25)


def find_energy_levels(n_max):
    """Нахождение E_n методом Брента для n=1..n_max."""
    E_min = PY0 * np.sqrt(D) * 1.001   # условие x0 > 1
    levels = []
    print('Уровни энергии E_n:')
    for n in range(1, n_max + 1):
        En = brentq(quant_equation, E_min, 500.0, args=(n,), xtol=1e-12)
        levels.append((n, En))
        print(f'  n={n}:  E_n = {En:.6f},  x0 = {turning_point(En):.3f}')
    return levels


# 2I(E) и дискретные уровни E_n
def fig2_quantization(E_levels, save='fig2_quantization.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(E_levels)))

    # Левая: 2I(E)
    E_max = E_levels[-1][1] * 1.1
    E_arr = np.linspace(PY0 * np.sqrt(D) * 1.002, E_max, 1000)
    two_I = [2 * I_analytical(e) for e in E_arr]

    ax1.plot(E_arr, two_I, color='royalblue', lw=2.2, label='$2I(E)$')
    for i, (n, En) in enumerate(E_levels):
        rhs = 2 * np.pi * H * (n + 0.25)
        ax1.axhline(rhs, color=colors[i], lw=1.2, ls='--', alpha=0.85)
        ax1.axvline(En,  color=colors[i], lw=1.2, ls=':',  alpha=0.85)
        ax1.plot(En, rhs, 'o', color=colors[i], ms=7, zorder=5)
        ax1.text(En + 0.05, rhs + 0.5,
                 f'$E_{n}={En:.2f}$', fontsize=8, color=colors[i])
    ax1.set_xlabel('$E$', fontsize=13)
    ax1.set_ylabel('$2I(E)$', fontsize=13)
    ax1.set_title('Графическое решение: $2I(E_n)=2\\pi h(n+1/4)$', fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Правая: горизонтальные уровни
    for i, (n, En) in enumerate(E_levels):
        ax2.axhline(En, color=colors[i], lw=2.2, xmin=0.1, xmax=0.88)
        ax2.text(0.90, En+0.15, f'$E_{{{n}}} = {En:.4f}$',
                 va='center', fontsize=10, color=colors[i])
    ax2.set_xlim(0, 1.3)
    ax2.set_ylim(0, E_levels[-1][1] * 1.12)
    ax2.set_xticks([])
    ax2.set_ylabel('$E_n$', fontsize=13)
    ax2.set_title(
        f'Дискретные уровни энергии $E_n$\n'
        f'$D={D},\\ p_y^0={PY0},\\ h={H}$', fontsize=11)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f'Сохранён: {save}')
    return fig



# ПРОКВАНТОВАННОЕ МНОГООБРАЗИЕ L^2_n И ПРОИЗВОДЯЩАЯ ФУНКЦИЯ S_n(x)

def action_S(x_val, En):
    return quad(lambda xp: np.sqrt(En**2 / (D * xp) - PY0**2),
                1.0, x_val)[0]


# ── Рисунок 3: семейство L^2_n + S_n(x) ───────────────────────────────────
def fig3_quantized_manifolds(E_levels, save='fig3_quantized_manifolds.png'):
    """
    Левая панель:  семейство L^2_n в проекции (x, p_x).
    Правая панель: производящая функция S_n(x) для каждого уровня.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x     = np.linspace(EPS, 1.0, 3000)
    x_s   = np.linspace(EPS, 1.0, 300)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(E_levels)))

    for i, (n, En) in enumerate(E_levels):
        px = px_upper(x, En)
        # Многообразие
        ax1.plot(x,  px, color=colors[i], lw=2.0, label=f'$n={n},\\ E_n={En:.2f}$')
        ax1.plot(x, -px, color=colors[i], lw=2.0)
        ax1.plot(1.0,  px_upper(np.array([1.0]), En)[0],
                 'o', color=colors[i], ms=6, zorder=5)
        ax1.plot(1.0, -px_upper(np.array([1.0]), En)[0],
                 'o', color=colors[i], ms=6, zorder=5)
        # Производящая функция
        S = np.array([action_S(xi, En) for xi in x_s])
        ax2.plot(x_s, S, color=colors[i], lw=2.0, label=f'$n={n},\\ E_n={En:.2f}$')

    for ax in [ax1, ax2]:
        ax.axvline(0, color='saddlebrown', lw=1.5, ls='--')
        ax.axvline(1, color='steelblue',   lw=1.5, ls='--')

    ax1.axhline(0, color='gray', lw=0.7, ls=':')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$p_x$', fontsize=13)
    ax1.set_title('Семейство проквантованных многообразий $L^2_n$', fontsize=12)
    ax1.legend(fontsize=8.5, loc='upper right')
    ax1.set_xlim(-0.05, 1.15)
    ax1.set_ylim(-150, 150)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel("$S_n(x)$", fontsize=13)
    ax2.set_title("Производящая функция  $S_n(x) = \\int_1^x p_x(x')\\,dx'$",
                  fontsize=12)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    #plt.show()
    plt.savefig(save, dpi=150)
    print(f'Сохранён: {save}')
    return fig


# ── Рисунок 4: 3D-поверхность L^2_1 в (x, y, p_x) ─────────────────────────
def fig4_manifold_3d(E_levels, save='fig4_manifold_3d.png'):
    """3D-поверхность первого проквантованного многообразия L^2_1."""
    n1, En1 = E_levels[0]
    x_3d = np.linspace(EPS, 1.0, 200)
    y_3d = np.linspace(-3.0, 3.0, 80)
    X, Y = np.meshgrid(x_3d, y_3d)
    PX   = px_upper(X, En1)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y,  PX, color='royalblue', alpha=0.6,
                    label='верхняя ветвь $p_x>0$')
    ax.plot_surface(X, Y, -PX, color='tomato',    alpha=0.4,
                    label='нижняя ветвь $p_x<0$')
    ax.set_xlabel('$x$',   fontsize=11, labelpad=8)
    ax.set_ylabel('$y$',   fontsize=11, labelpad=8)
    ax.set_zlabel('$p_x$', fontsize=11, labelpad=8)
    ax.set_title(f'$L^2_1$ в пространстве $(x, y, p_x)$,  $E_1={En1:.3f}$',
                 fontsize=12)
    ax.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f'Сохранён: {save}')
    return fig


# ШАГ 4.  КАНОНИЧЕСКИЙ ОПЕРАТОР МАСЛОВА
def G_generating(px, En):
    """
    Производящая функция G(p_x) в p_x-карте.
    """
    return (En**2 / (D * PY0)) * np.arctan(px / PY0)


def amplitude_A(px, En):
    """
    Амплитуда канонического оператора:
    """
    return np.sqrt(2 * En**2 * np.abs(px) / (D * (px**2 + PY0**2)**2))


def maslov_phase(px):
    """
    Фаза Маслова:
      px > 0: верхняя ветвь, ещё не прошли берег  => phi_M = 0
      px < 0: нижняя ветвь, берег пройден (Δm=1)  => phi_M = −pi/2
    """
    return np.where(px >= 0, 0.0, -np.pi / 2)


def u_tilde(px, En):
    """
    Канонический оператор K^h_{L^2_n}[1] в p_x-карте:
      u~(px) = A(px) * exp(i/h * G(px) + i*phi_M(px))
    """
    phi = G_generating(px, En) / H + maslov_phase(px)
    return amplitude_A(px, En) * np.exp(1j * phi)


def u_wave(x_val, En, P_max=80, N=20000):
    """
    Волновая функция через обратное преобразование Фурье.
    Вычисляется численно методом трапеций.
    """
    px        = np.linspace(-P_max, P_max, N)
    integrand = u_tilde(px, En) * np.exp(1j * x_val * px / H)
    return np.trapezoid(integrand, px) / np.sqrt(2 * np.pi * H)


# u~(p_x) и волновые функции u(x)
def fig5_canonical_operator(E_levels, x_arr, save='fig5_canonical_operator.png'):
    """
    Панель 2×2:
      (0,0) амплитуда A(px)                 для n=1
      (0,1) Re и Im u~(px) в p_x-карте       для n=1
      (1,0) Re[u(x)] — волновая функция      для n=1..4
      (1,1) |u(x)|  — амплитуда колебаний    для n=1..4
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors  = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    px_plot = np.linspace(-15, 15, 3000)

    # ── Верхние панели: u~ для n=1 ──────────────────────────
    n1, En1 = E_levels[0]
    ut      = u_tilde(px_plot, En1)

    axes[0, 0].plot(px_plot, np.abs(ut), color='royalblue', lw=2.0)
    axes[0, 0].axvline(0, color='gray', lw=0.8, ls=':')
    axes[0, 0].set_title(f'Амплитуда $A(p_x)$,  $n=1,\\ E_1={En1:.3f}$',
                         fontsize=11)
    axes[0, 0].set_xlabel('$p_x$')
    axes[0, 0].set_ylabel('$A(p_x)$')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(px_plot, np.real(ut), color='royalblue', lw=1.5,
                    label=r'$\mathrm{Re}[\tilde{u}]$', alpha=0.85)
    axes[0, 1].plot(px_plot, np.imag(ut), color='tomato',    lw=1.5,
                    label=r'$\mathrm{Im}[\tilde{u}]$', alpha=0.85)
    axes[0, 1].axvline(0, color='gray', lw=0.8, ls=':')
    axes[0, 1].set_title(r'$\tilde{u}(p_x)$ в $p_x$-карте,  $n=1$',
                         fontsize=11)
    axes[0, 1].set_xlabel('$p_x$')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    for i, (n, En) in enumerate(E_levels[:4]):
        u_vals = np.array([u_wave(x, En) for x in x_arr])
        axes[1, 0].plot(x_arr, np.real(u_vals), color=colors[i], lw=1.8, label=f'$n={n},\\ E_n={En:.2f}$')
        axes[1, 1].plot(x_arr, np.abs(u_vals),  color=colors[i], lw=1.8, label=f'$n={n},\\ E_n={En:.2f}$')

    for ax, title, yl in [
        (axes[1, 0], r'$\mathrm{Re}[u(x)]$  — волновая функция',
                     r'$\mathrm{Re}[u(x)]$'),
        (axes[1, 1], r'$|u(x)|$  — амплитуда колебаний',
                     r'$|u(x)|$'),
    ]:
        ax.axvline(1, color='steelblue',   lw=1.5, ls='--', label='стенка $x=1$')
        ax.axvline(0, color='saddlebrown', lw=1.5, ls='--', label='берег $x=0$')
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel(yl, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save, dpi=150)
    plt.show()
    print(f'Сохранён: {save}')
    return fig


if __name__ == '__main__':
    E_test = 2.0
    print(f'Проверка I(E) при E={E_test}:')
    print(f'  аналитически = {I_analytical(E_test):.8f}')
    print(f'  численно     = {I_numerical(E_test):.8f}\n')

    E_levels = find_energy_levels(N_MAX)
    print()

    x_arr = np.linspace(EPS, 1.0, 150)

    fig1_manifold(E_single=1.0)
    fig2_quantization(E_levels)
    fig3_quantized_manifolds(E_levels)
    fig4_manifold_3d(E_levels)
    fig5_canonical_operator(E_levels, x_arr)