import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


def quadratic(x: float) -> float:
    return x ** 2

def derivative_quadratic(x: float) -> float:
    return 2 * x

def double_derivative_quadratic(x: float) -> float:
    return 2


class RiemannIntegral:
    def __init__(self, integral_sum: float = 7 / 3,
                 func = quadratic, derivative = derivative_quadratic, double_derivative = double_derivative_quadratic,
                 left: int = 1, right: int = 2, number_of_dots: int = 30) -> None:
        self.real_sum = integral_sum
        self.func = func
        self.der = derivative
        self.double_der = double_derivative
        self.left = left
        self.right = right
        self.number_of_dots = number_of_dots
        self.mult = (right - left) / self.number_of_dots
        self.x_dots = np.array([(left + k * self.mult) for k in range(self.number_of_dots + 1)])

        self.rec_error_rate = max(abs(self.der(x)) for x in self.x_dots) * self.mult * (right - left) / 2
        self.mid_rec_error_rate = max(abs(self.double_der(x)) for x in self.x_dots) * pow(self.mult, 2) * (right - left) / 24
        self.trap_error_rate = max(abs(self.double_der(x)) for x in self.x_dots) * pow(self.mult, 2) * (right - left) / 12

    def print_sum(self, sum_type: str, func_sum: float, real_error: float) -> None:
        print(f'Расчёт интегральной суммы методом {sum_type}')
        print(f'Интегральная сумма равна: {round(func_sum, 5)}')
        print(f'Погрешность вычислений: {round(abs(self.real_sum - func_sum), 5)}')
        print(f'Допустимая погрешность: {round(real_error, 5)}')
        print()

    def right_rectangle(self):
        y_dots = np.array([self.func(x) for x in self.x_dots][1:])
        integral_sum = sum(y_dots) * self.mult

        self.print_sum('правых прямоугольников', integral_sum, self.rec_error_rate)
        self.plot_riemann(y_dots, 'rectangle', 'Правые прямоугольники функции f(x)')

    def left_rectangle(self):
        y_dots = np.array([self.func(x) for x in self.x_dots][:-1])
        integral_sum = sum(y_dots) * self.mult

        self.print_sum('левых прямоугольников', integral_sum, self.rec_error_rate)
        self.plot_riemann(y_dots, 'rectangle', 'Левые прямоугольники функции f(x)')

    def mid_rectangle(self):
        sum_series_function = lambda x, y: self.func((self.x_dots[x] + self.x_dots[y]) / 2)

        y_mid = np.array([sum_series_function(n - 1, n) for n in range(1, len(self.x_dots))])
        integral_sum = sum(y_mid) * self.mult

        self.print_sum('средних прямоугольников', integral_sum, self.mid_rec_error_rate)
        self.plot_riemann(y_mid, 'rectangle', 'Средние прямоугольники функции f(x)')

    def trapezoid(self):
        sum_series_function = lambda x, y:(self.func(self.x_dots[x]) + self.func(self.x_dots[y])) / 2

        y_dots = np.array([self.func(x) for x in self.x_dots])
        y_mid = np.array([sum_series_function(n - 1, n) for n in range(1, len(self.x_dots))])
        integral_sum = sum(y_mid) * self.mult

        self.print_sum('трапеций', integral_sum, self.trap_error_rate)
        self.plot_riemann(y_dots, 'trapezoid', 'Трапеции функции f(x)')

    def plot_riemann(self, dots, riemann_type: str, title: str):
        plt.style.use('seaborn-v0_8-colorblind')

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.text(1, 1, 'FF')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

        dots_func = self.number_of_dots * 10
        mult_func = (self.right - self.left) / dots_func
        x_func = [(self.left + k * mult_func) for k in range(dots_func + 1)]
        y_func = [self.func(x) for x in x_func]
        func_graphic = ax.plot(x_func, y_func, color='k', linewidth=2, label='f(x)', zorder=3)

        if riemann_type == 'rectangle':
            for i in range(len(dots)):
                color = '#4FF01D' if dots[i] > 0 else '#FA335B'
                rect = Rectangle((self.x_dots[i], 0), width=(self.x_dots[i + 1] - self.x_dots[i]), height=dots[i],
                                 edgecolor='black', facecolor=color, linewidth=0.7, zorder=2)
                ax.add_patch(rect)
        else:
            for i in range(1, len(dots)):
                color = '#4FF01D' if dots[i] > 0 else '#FA335B'
                x = (self.x_dots[i - 1], self.x_dots[i], self.x_dots[i], self.x_dots[i - 1])
                y = (0, 0, dots[i], dots[i - 1])
                trap = Polygon(xy=list(zip(x,y)),
                               edgecolor='black', facecolor=color, linewidth=0.7, zorder=2)
                ax.add_patch(trap)
        plt.legend(loc='best', fancybox=True)
        plt.grid(True, zorder=0)
        plt.savefig(f'output/{title}.png', dpi=700)
        plt.close()
