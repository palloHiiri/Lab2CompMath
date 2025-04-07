import numpy as np
import matplotlib.pyplot as plt
from math import isclose, sin, cos, exp, log, tan, sqrt


class NonlinearEquationSolver:
    def __init__(self):
        self.equations = {
            1: {'func': lambda x: x ** 3 - 2 * x - 5, 'desc': 'x^3 - 2x - 5 = 0'},
            2: {'func': lambda x: sin(x) + 0.5 * x - 1, 'desc': 'sin(x) + 0.5x - 1 = 0'},
            3: {'func': lambda x: exp(-x) - x, 'desc': 'e^(-x) - x = 0'},
            4: {'func': lambda x: log(x) + x - 2, 'desc': 'ln(x) + x - 2 = 0'},
            5: {'func': lambda x: x ** 2 - cos(x), 'desc': 'x^2 - cos(x) = 0'}
        }

        self.systems = {
            1: {
                'funcs': [
                    lambda x, y: x ** 2 + y ** 2 - 4,
                    lambda x, y: x ** 2 - y - 1
                ],
                'desc': 'x^2 + y^2 = 4\nx^2 - y = 1',
                'jacobian': lambda x, y: np.array([
                    [2 * x, 2 * y],
                    [2 * x, -1]
                ])
            },
            2: {
                'funcs': [
                    lambda x, y: sin(x) + y - 1.2,
                    lambda x, y: 2 * x + cos(y) - 2
                ],
                'desc': 'sin(x) + y = 1.2\n2x + cos(y) = 2',
                'jacobian': lambda x, y: np.array([
                    [cos(x), 1],
                    [2, -sin(y)]
                ])
            },
            3: {
                'funcs': [
                    lambda x, y: x ** 2 + y - 3,
                    lambda x, y: x + y ** 2 - 2
                ],
                'desc': 'x^2 + y = 3\nx + y^2 = 2',
                'jacobian': lambda x, y: np.array([
                    [2 * x, 1],
                    [1, 2 * y]
                ])
            }
        }

        self.max_iterations = 1000

    def choose_input_method(self):
        print("\nВыберите способ ввода данных:")
        print("1 - С клавиатуры")
        print("2 - Из файла")
        while True:
            choice = input("Ваш выбор (1/2): ")
            if choice in ['1', '2']:
                return choice
            print("Режимов два. Выбирай из предложенных")

    def read_equation_from_keyboard(self):
        print("\nДоступные уравнения:")
        for num, eq in self.equations.items():
            print(f"{num}. {eq['desc']}")

        while True:
            try:
                eq_num = int(input("\nВыберите номер уравнения: "))
                if eq_num not in self.equations:
                    raise ValueError
                break
            except ValueError:
                print("Уравнений всего пять. Выбирай из предложенных")

        while True:
            try:
                a = float(input("Введите левую границу интервала: ").replace(',', '.'))
                b = float(input("Введите правую границу интервала: ").replace(',', '.'))
                if a >= b:
                    print("Таки левая левее правой должна быть")
                    continue
                break
            except ValueError:
                print("Числа хочу... числа...")

        while True:
            try:
                eps = float(input("Введите точность вычисления: ").replace(',', '.'))
                if eps <= 0:
                    print("Хочу точность больше нуля! Больше нуля хочу!!!")
                    continue
                elif eps >1:
                    print("А можно точность поменьше плись(")
                    continue
                break
            except ValueError:
                print("Вводим нормальную точность, нормальную!")

        return eq_num, a, b, eps

    def read_equation_from_file(self):
        print("\nДоступные уравнения:")
        for num, eq in self.equations.items():
            print(f"{num}. {eq['desc']}")
        while True:
            try:
                filename = input("Введите имя файла: ")
                with open(filename, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]

                if len(lines) < 4:
                    raise ValueError("Строк должно быть минимум 4. Куда строки дел?")

                eq_num = int(lines[0])
                if eq_num not in self.equations:
                    raise ValueError("Таких уравнений у нас нет")

                a = float(lines[1].replace(',', '.'))
                b = float(lines[2].replace(',', '.'))
                if a >= b:
                    raise ValueError("Таки левая граница должна быть левее правой")

                eps = float(lines[3].replace(',', '.'))
                if eps <= 0:
                    raise ValueError("Ненене, слишком маленькая точность!")
                elif eps > 1:
                    raise ValueError("А можно поменьше плись(")

                return eq_num, a, b, eps

            except FileNotFoundError:
                print("Такого файла в наличии нет, попробуйте позже")
            except ValueError as e:
                print(f"Ошибка в данных файла: {e}. Давай по новой!")
            except Exception as e:
                print(f"Хрень какая-то: {e}. Давай по новой!")

    def read_system_from_keyboard(self):
        print("\nДоступные системы уравнений:")
        for num, sys in self.systems.items():
            print(f"{num}. \n{sys['desc']}\n")

        while True:
            try:
                sys_num = int(input("\nВыберите номер системы: "))
                if sys_num not in self.systems:
                    raise ValueError
                break
            except ValueError:
                print("Число. Число от 1 до 3. Так сложно чтоле?")

        while True:
            try:
                x0 = float(input("Введите начальное приближение для x: ").replace(',', '.'))
                y0 = float(input("Введите начальное приближение для y: ").replace(',', '.'))
                break
            except ValueError:
                print("Числаааааа, числааааааа хочу!")

        while True:
            try:
                eps = float(input("Введите точность вычисления: ").replace(',', '.'))
                if eps <= 0:
                    print("Точность должна быть положительным числом!")
                    continue
                elif eps > 1:
                    print("А можно точность поменьше плись(")
                    continue
                break
            except ValueError:
                print("Дайте мне адекватную точность. А то работать не буду")

        return sys_num, x0, y0, eps

    def read_system_from_file(self):
        print("\nДоступные системы уравнений:")
        for num, sys in self.systems.items():
            print(f"{num}. \n{sys['desc']}\n")
        while True:
            try:
                filename = input("Введите имя файла: ")
                with open(filename, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]

                if len(lines) < 4:
                    raise ValueError("Строк не хватает. Надо минимум 4. Дай мне строк!")

                sys_num = int(lines[0])
                if sys_num not in self.systems:
                    raise ValueError("Такой системы у нас нет, зайдите позже или поменяйте данные")

                x0 = float(lines[1].replace(',', '.'))
                y0 = float(lines[2].replace(',', '.'))
                eps = float(lines[3].replace(',', '.'))
                if eps <= 0:
                    raise ValueError("слишком малаааааа! Точность бы побольше чутка, мы такое не потянем")
                elif eps > 1:
                    raise ValueError("А можно точность поменьше плись(")

                return sys_num, x0, y0, eps

            except FileNotFoundError:
                print("Такого файла у нас нет. Возвращайтесь позже")
            except ValueError as e:
                print(f"Ошибка в данных файла: {e}. Давай по новой!")
            except Exception as e:
                print(f"Все фигня: {e}. Давай по новой!")

    def verify_interval(self, func, a, b):
        try:
            fa = func(a)
            fb = func(b)
        except ValueError as e:
            return False, f"Ошибка вычисления функции: {e}"
        except Exception as e:
            return False, f"Неизвестная ошибка при вычислении функции: {e}"

        if isclose(fa, 0, abs_tol=1e-9):
            return True, f"Найдено точное решение на левой границе x = {a}"
        if isclose(fb, 0, abs_tol=1e-9):
            return True, f"Найдено точное решение на правой границе x = {b}"

        if fa * fb > 0:
            x = np.linspace(a, b, 100)
            y = [func(xi) for xi in x]
            sign_changes = 0
            for i in range(len(y) - 1):
                if y[i] * y[i + 1] <= 0:
                    sign_changes += 1

            if sign_changes == 0:
                return False, "Функция не меняет знак на интервале (корней нет)"
            elif sign_changes > 1:
                return False, f"Функция меняет знак {sign_changes} раз(a) - у несколько корней. Хьюстон, для меня слишком сложно"

        return True, "Интервал корректен"

    def plot_function(self, func, a, b, root=None):
        x = np.linspace(a, b, 400)
        y = [func(xi) for xi in x]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Функция')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        if root is not None:
            plt.scatter([root], [func(root)], color='red', label=f'Корень: {root:.5f}')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('График функции')
        plt.grid()
        plt.legend()
        plt.show()

    def chord_method(self, func, a, b, eps):
        fa = func(a)
        fb = func(b)
        iterations = 0

        while abs(b - a) > eps and iterations < self.max_iterations:
            c = a - fa * (b - a) / (fb - fa)
            fc = func(c)

            if isclose(fc, 0, abs_tol=eps):
                return c, fc, iterations + 1

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            iterations += 1

        root = (a + b) / 2
        return root, func(root), iterations

    def secant_method(self, func, a, b, eps):
        x0 = a
        x1 = b
        f0 = func(x0)
        f1 = func(x1)
        iterations = 0

        while abs(x1 - x0) > eps and iterations < self.max_iterations:
            if isclose(f1 - f0, 0, abs_tol=1e-12):
                break

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, x1 = x1, x2
            f0, f1 = f1, func(x2)
            iterations += 1

        return x1, f1, iterations

    def simple_iteration_method(self, func, a, b, eps):
        lambda_choices = [-0.1, 0.1,
                          -1 / max(abs(func(a)), abs(func(b))) if max(abs(func(a)), abs(func(b))) != 0 else -0.1]

        for lambda_ in lambda_choices:
            phi = lambda x: x + lambda_ * func(x)

            h = (b - a) / 100
            max_derivative = 0
            for x in np.arange(a, b, h):
                try:
                    derivative = abs((phi(x + h) - phi(x)) / h)
                    if derivative > max_derivative:
                        max_derivative = derivative
                except:
                    continue

            if max_derivative >= 1:
                continue

            x0 = (a + b) / 2
            x1 = phi(x0)
            iterations = 1

            while abs(x1 - x0) > eps and iterations < self.max_iterations:
                x0 = x1
                x1 = phi(x0)
                iterations += 1

            return x1, func(x1), iterations

        print("Предупреждение: не удалось найти преобразование с гарантированной сходимостью")
        lambda_ = -0.1
        phi = lambda x: x + lambda_ * func(x)
        x0 = (a + b) / 2
        x1 = phi(x0)
        iterations = 1

        while abs(x1 - x0) > eps and iterations < self.max_iterations:
            x0 = x1
            x1 = phi(x0)
            iterations += 1

        return x1, func(x1), iterations

    def newton_method_system(self, system_num, x0, y0, eps):
        system = self.systems[system_num]
        f1, f2 = system['funcs']
        jacobian = system['jacobian']

        x = np.array([x0, y0], dtype=float)
        iterations = 0
        errors = []

        while True:
            F = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
            J = jacobian(x[0], x[1])

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                print("Ошибка: матрица Якоби вырождена")
                return None, None, None, None

            x_new = x + delta
            error = np.linalg.norm(x_new - x)
            errors.append(error)
            iterations += 1

            if error < eps or iterations >= self.max_iterations:
                break

            x = x_new

        residual1 = f1(x[0], x[1])
        residual2 = f2(x[0], x[1])
        residuals = np.array([residual1, residual2])

        return x, residuals, iterations, errors

    def plot_system(self, system_num, solution=None):
        system = self.systems[system_num]
        f1, f2 = system['funcs']

        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)

        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z1[i, j] = f1(X[i, j], Y[i, j])
                Z2[i, j] = f2(X[i, j], Y[i, j])

        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z1, levels=[0], colors='r', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='b', linewidths=2)

        if solution is not None:
            plt.scatter([solution[0]], [solution[1]], color='green', s=100, label='Решение')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'График системы уравнений\n{system["desc"]}')
        plt.grid()
        plt.show()

    def choose_output_method(self):
        print("\nВыберите способ вывода результатов:")
        print("1 - На экран")
        print("2 - В файл")
        while True:
            choice = input("Ваш выбор (1/2): ")
            if choice in ['1', '2']:
                return choice
            print("Да ну пора уже научиться выбирать из двух вариантов")

    def output_to_screen(self, result, eq_desc):
        root, f_root, iterations = result
        print("\nРезультаты решения уравнения:")
        print(f"Уравнение: {eq_desc}")
        print(f"Найденный корень: {root:.10f}")
        print(f"Значение функции в корне: {f_root:.2e}")
        print(f"Количество итераций: {iterations}")

    def output_to_file(self, result, eq_desc, filename="output.txt"):
        root, f_root, iterations = result
        with open(filename, 'w') as f:
            f.write("Результаты решения уравнения:\n")
            f.write(f"Уравнение: {eq_desc}\n")
            f.write(f"Найденный корень: {root:.10f}\n")
            f.write(f"Значение функции в корне: {f_root:.2e}\n")
            f.write(f"Количество итераций: {iterations}\n")
        print(f"Результаты сохранены в файл {filename}")

    def output_system_results(self, solution, residuals, iterations, errors, system_desc):
        print("\nРезультаты решения системы уравнений:")
        print(f"Система уравнений:\n{system_desc}")
        print(f"Найденное решение: x = {solution[0]:.8f}, y = {solution[1]:.8f}")
        print(f"Невязки уравнений: {residuals[0]:.2e}, {residuals[1]:.2e}")
        print(f"Количество итераций: {iterations}")

        print("\nВектор погрешностей на последних итерациях:")
        for i, error in enumerate(errors[-5:], start=max(1, len(errors) - 4)):
            print(f"Итерация {i}: {error:.2e}")

    def run(self):
        print("Программа для решения нелинейных уравнений и систем")

        while True:
            print("\nВыберите режим работы:")
            print("1 - Решение нелинейного уравнения")
            print("2 - Решение системы нелинейных уравнений")
            print("3 - Выход")

            mode = input("Ваш выбор (1/2/3): ")

            if mode == '1':
                self.solve_equation()
            elif mode == '2':
                self.solve_system()
            elif mode == '3':
                break
            else:
                print("Учимся выбирать из предложенных вариантов")

    def solve_equation(self):
        input_choice = self.choose_input_method()

        if input_choice == '1':
            eq_num, a, b, eps = self.read_equation_from_keyboard()
        else:
            eq_num, a, b, eps = self.read_equation_from_file()

        func = self.equations[eq_num]['func']
        eq_desc = self.equations[eq_num]['desc']

        valid, message = self.verify_interval(func, a, b)
        print(f"\nПроверка интервала: {message}")
        if not valid:
            return

        self.plot_function(func, a, b)

        print("\nВыберите метод решения:")
        print("1 - Метод хорд")
        print("2 - Метод секущих")
        print("3 - Метод простых итераций")
        while True:
            method_choice = input("Ваш выбор (1/2/3): ")
            if method_choice in ['1', '2', '3']:
                break
            print("Выбираем из предложенных вариантов, не тупим")

        if method_choice == '1':
            result = self.chord_method(func, a, b, eps)
        elif method_choice == '2':
            result = self.secant_method(func, a, b, eps)
        elif method_choice == '3':
            result = self.simple_iteration_method(func, a, b, eps)

        output_choice = self.choose_output_method()
        if output_choice == '1':
            self.output_to_screen(result, eq_desc)
        else:
            self.output_to_file(result, eq_desc)

        self.plot_function(func, a, b, result[0])

    def solve_system(self):
        input_choice = self.choose_input_method()

        if input_choice == '1':
            sys_num, x0, y0, eps = self.read_system_from_keyboard()
        else:
            sys_num, x0, y0, eps = self.read_system_from_file()

        self.plot_system(sys_num)

        solution, residuals, iterations, errors = self.newton_method_system(
            sys_num, x0, y0, eps
        )

        if solution is not None:
            self.output_system_results(
                solution, residuals, iterations, errors,
                self.systems[sys_num]['desc']
            )
            self.plot_system(sys_num, solution)


if __name__ == "__main__":
    solver = NonlinearEquationSolver()
    solver.run()