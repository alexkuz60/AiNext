import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.special import comb

# Параметры
max_num = 50  # Максимальное значение в диапазоне (от 1 до max_num)
input_arr_size = 750  # Размер входного массива

# Функция для вычисления теоретических вероятностей совпадений
def calculate_theoretical_probabilities(n_columns, max_num):
    """
    Вычисляет теоретические вероятности совпадения k чисел из n_columns
    при случайном выборе n_columns чисел из max_num без повторений.
    
    Возвращает массив вероятностей для k = 0, 1, 2, ..., n_columns
    """
    probabilities = []
    
    # Общее количество возможных комбинаций
    total_combinations = comb(max_num, n_columns)
    
    for k in range(n_columns + 1):
        # Вероятность совпадения ровно k чисел
        # Используем гипергеометрическое распределение
        # P(X = k) = C(n_columns, k) * C(max_num - n_columns, n_columns - k) / C(max_num, n_columns)
        if n_columns <= max_num:
            prob = (comb(n_columns, k) * comb(max_num - n_columns, n_columns - k)) / total_combinations
        else:
            # Если n_columns > max_num, это невозможная ситуация
            prob = 0.0 if k < max_num else 1.0
        
        probabilities.append(prob)
    
    return np.array(probabilities)

# Функция для загрузки данных из CSV файла
def load_data_from_csv(file_path):
    """
    Загружает данные из CSV файла.
    Предполагается, что файл содержит n_columns столбцов с числами.
    """
    try:
        df = pd.read_csv(file_path, delimiter='\t', skip_blank_lines=True, skiprows=1)
        n_columns = len(df.columns)
        print(f"Загружено {len(df)} строк с {n_columns} столбцами.")
        return df, n_columns
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        # Создаем случайные данные для демонстрации, если файл не найден
        print("Создание демонстрационных данных...")
        n_columns = 3  # Пример: 3 столбца
        
        # Создаем DataFrame с n_columns столбцами без повторений в строках
        data = []
        for _ in range(input_arr_size):
            # Генерируем n_columns уникальных чисел от 1 до max_num
            row = np.random.choice(range(1, max_num+1), size=n_columns, replace=False)
            data.append(row)
        
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(n_columns)])
        return df, n_columns

# Функция для создания матрицы переходов с учетом ограничений
def create_transition_matrices(data, n_columns, max_num):
    """
    Создает n_columns-1 матриц переходов.
    Каждая матрица показывает вероятность перехода от числа в столбце i к числу в столбце i+1,
    учитывая, что числа не повторяются в строке.
    """
    transition_matrices = []
    
    for i in range(n_columns - 1):
        # Матрица переходов от столбца i к столбцу i+1
        transition_matrix = np.zeros((max_num, max_num))
        
        col_current = data.iloc[:, i].values
        col_next = data.iloc[:, i+1].values
        
        for j in range(len(data)):
            current_state = col_current[j] - 1  # Индексы от 0 до max_num-1
            next_state = col_next[j] - 1
            transition_matrix[current_state, next_state] += 1
        
        # Нормализуем строки для получения вероятностей
        row_sums = transition_matrix.sum(axis=1)
        for j in range(max_num):
            if row_sums[j] > 0:
                transition_matrix[j, :] /= row_sums[j]
        
        transition_matrices.append(transition_matrix)
    
    return transition_matrices

# Функция для генерации следующего числа с учетом уже выбранных чисел
def generate_next_number(current_state, transition_matrix, excluded_numbers, max_num):
    """
    Генерирует следующее число на основе текущего состояния и матрицы переходов,
    исключая числа, которые уже были выбраны в текущей строке.
    """
    probabilities = transition_matrix[current_state - 1].copy()
    
    # Устанавливаем вероятность 0 для исключенных чисел
    for num in excluded_numbers:
        if 1 <= num <= max_num:
            probabilities[num - 1] = 0
    
    # Если все вероятности равны 0, выбираем из оставшихся чисел равновероятно
    if np.sum(probabilities) == 0:
        available_numbers = [i for i in range(1, max_num + 1) if i not in excluded_numbers]
        if available_numbers:
            return np.random.choice(available_numbers)
        else:
            return None  # Если все числа исключены
    
    # Нормализуем вероятности
    probabilities = probabilities / np.sum(probabilities)
    
    return np.random.choice(range(1, max_num + 1), p=probabilities)

# Функция для генерации полной строки
def generate_row(transition_matrices, n_columns, max_num, first_number=None):
    """
    Генерирует строку из n_columns уникальных чисел.
    """
    row = np.zeros(n_columns, dtype=int)
    
    # Если первое число не задано, выбираем случайно
    if first_number is None:
        row[0] = np.random.randint(1, max_num + 1)
    else:
        row[0] = first_number
    
    excluded_numbers = [row[0]]
    
    for i in range(1, n_columns):
        current_state = row[i-1]
        transition_matrix = transition_matrices[i-1]
        
        next_number = generate_next_number(current_state, transition_matrix, excluded_numbers, max_num)
        
        if next_number is None:
            # Если не удалось сгенерировать число (маловероятно), выбираем случайно
            available_numbers = [j for j in range(1, max_num + 1) if j not in excluded_numbers]
            if available_numbers:
                next_number = np.random.choice(available_numbers)
            else:
                # Если все числа использованы, это ошибка в логике
                raise ValueError("Не удалось сгенерировать уникальное число")
        
        row[i] = next_number
        excluded_numbers.append(next_number)
    
    return row

# Основная функция
def main(file_path="random_numbers.csv"):
    # Загрузка данных
    data, n_columns = load_data_from_csv('5_50.txt')
    
    # Создание матриц переходов
    transition_matrices = create_transition_matrices(data, n_columns, max_num)
    
    # Проверка результатов
    comparison_results = []
    match_counts = []  # Для графика благоприятности прогноза
    
    for i in range(len(data)):
        original_row = data.iloc[i].values
        
        # Генерируем строку, начиная с первого числа оригинальной строки
        generated_row = generate_row(transition_matrices, n_columns, max_num, first_number=original_row[0])
        
        # Подсчитываем количество совпадений
        matches = sum(original_row[j] == generated_row[j] for j in range(n_columns))
        match_counts.append(matches)
        
        comparison_results.append({
            'original': original_row,
            'generated': generated_row,
            'matches': matches
        })
    
    # Анализ результатов
    total_elements = len(data) * n_columns
    total_matches = sum(match_counts)
    match_percentage = (total_matches / total_elements) * 100
    
    print(f"\nРезультаты:")
    print(f"Всего элементов: {total_elements}")
    print(f"Всего совпадений: {total_matches}")
    print(f"Процент совпадений: {match_percentage:.2f}%")
    
    # Расчет теоретических вероятностей
    theoretical_probs = calculate_theoretical_probabilities(n_columns, max_num)
    
    # Расчет фактических частот совпадений
    match_counts_hist = np.zeros(n_columns + 1, dtype=int)
    for count in match_counts:
        match_counts_hist[count] += 1
    
    # Преобразуем в вероятности
    actual_probs = match_counts_hist / len(match_counts)
    
    # Расчет отношения фактических вероятностей к теоретическим
    ratio_to_theoretical = np.zeros(n_columns + 1)
    for i in range(n_columns + 1):
        if theoretical_probs[i] > 0:
            ratio_to_theoretical[i] = actual_probs[i] / theoretical_probs[i]
        else:
            ratio_to_theoretical[i] = float('inf') if actual_probs[i] > 0 else 0
    
    # Вывод сравнения вероятностей
    print("\nСравнение вероятностей совпадений:")
    print("Кол-во совпадений | Теоретическая вероятность | Фактическая вероятность | Отношение")
    print("-" * 80)
    for i in range(n_columns + 1):
        print(f"{i:^16} | {theoretical_probs[i]:^25.6f} | {actual_probs[i]:^23.6f} | {ratio_to_theoretical[i]:^9.2f}")
    
    # Визуализация результатов
    plt.figure(figsize=(15, 15))
    
    # 1. График распределения совпадений
    plt.subplot(3, 2, 1)
    plt.hist(match_counts, bins=range(n_columns + 2), alpha=0.7, color='blue')
    plt.title('Распределение количества совпадений по строкам')
    plt.xlabel('Количество совпадений')
    plt.ylabel('Частота')
    plt.xticks(range(n_columns + 1))
    
    # 2. График благоприятности прогноза (совпадения по итерациям)
    plt.subplot(3, 2, 2)
    plt.plot(range(len(match_counts)), match_counts, 'o-', alpha=0.6)
    plt.title('Благоприятность прогноза по итерациям')
    plt.xlabel('Номер итерации')
    plt.ylabel('Количество совпадений')
    plt.axhline(y=np.mean(match_counts), color='r', linestyle='--', label=f'Среднее: {np.mean(match_counts):.2f}')
    plt.legend()
    
    # 3. Тепловая карта первой матрицы переходов
    plt.subplot(3, 2, 3)
    sns.heatmap(transition_matrices[0], annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Матрица переходов: столбец 1 -> столбец 2')
    plt.xlabel('Следующее число')
    plt.ylabel('Текущее число')
    
    # 4. Сравнение распределений по столбцам
    plt.subplot(3, 2, 4)
    
    # Создаем массивы для оригинальных и сгенерированных данных по столбцам
    original_by_column = [data.iloc[:, j].values for j in range(n_columns)]
    generated_by_column = [np.array([comparison_results[i]['generated'][j] for i in range(len(data))]) 
                          for j in range(n_columns)]
    
    # Отображаем распределение для первого столбца
    column_idx = 0  # Можно изменить для отображения других столбцов
    plt.hist(original_by_column[column_idx], bins=range(1, max_num + 2), 
             alpha=0.7, label=f'Оригинал (столбец {column_idx+1})')
    plt.hist(generated_by_column[column_idx], bins=range(1, max_num + 2), 
             alpha=0.5, label=f'Сгенерировано (столбец {column_idx+1})')
    plt.title(f'Сравнение распределений для столбца {column_idx+1}')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.legend()
    
    # 5. Сравнение теоретических и фактических вероятностей
    plt.subplot(3, 2, 5)
    x = np.arange(n_columns + 1)
    width = 0.35
    
    plt.bar(x - width/2, theoretical_probs, width, label='Теоретическая', alpha=0.7, color='blue')
    plt.bar(x + width/2, actual_probs, width, label='Фактическая', alpha=0.7, color='orange')
    
    plt.title('Сравнение теоретических и фактических вероятностей совпадений')
    plt.xlabel('Количество совпадений')
    plt.ylabel('Вероятность')
    plt.xticks(x)
    plt.legend()
    
    # 6. Отношение фактических вероятностей к теоретическим
    plt.subplot(3, 2, 6)
    plt.bar(x, ratio_to_theoretical, color='green', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Теоретическое ожидание')
    
    plt.title('Отношение фактических вероятностей к теоретическим')
    plt.xlabel('Количество совпадений')
    plt.ylabel('Отношение (факт/теория)')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('random_generator_analysis.png')
    plt.show()
    
    # Дополнительный график: скользящее среднее совпадений
    plt.figure(figsize=(12, 6))
    window_size = min(10, len(match_counts) // 10)  # Размер окна для скользящего среднего
    rolling_mean = pd.Series(match_counts).rolling(window=window_size).mean()
    
    # Теоретическое ожидание среднего числа совпадений
    theoretical_mean = sum(i * theoretical_probs[i] for i in range(n_columns + 1))
    
    plt.plot(range(len(match_counts)), match_counts, 'o', alpha=0.3, label='Совпадения')
    plt.plot(range(len(rolling_mean)), rolling_mean, 'r-', linewidth=2, 
             label=f'Скользящее среднее (окно={window_size})')
    plt.axhline(y=theoretical_mean, color='g', linestyle='--', 
                label=f'Теоретическое среднее: {theoretical_mean:.2f}')
    
    plt.title('Динамика совпадений с течением времени')
    plt.xlabel('Номер итерации')
    plt.ylabel('Количество совпадений')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('match_dynamics.png')
    plt.show()
    
    # Дополнительный график: эффективность ГСЧ по сравнению с теоретическими вероятностями
    plt.figure(figsize=(14, 8))
    
    # Создаем массив для хранения отношений для каждой итерации
    iteration_effectiveness = []
    
    # Для каждой итерации вычисляем, во сколько раз вероятность полученного результата
    # отличается от теоретической вероятности
    for matches in match_counts:
        if theoretical_probs[matches] > 0:
            # Вероятность получить exactly matches совпадений
            effectiveness = 1.0 / theoretical_probs[matches]
            iteration_effectiveness.append(effectiveness)
        else:
            # Если теоретическая вероятность равна 0, это "бесконечно эффективно"
            iteration_effectiveness.append(float('inf'))
    
    # Ограничиваем очень большие значения для лучшей визуализации
    max_display_value = 100
    capped_effectiveness = [min(e, max_display_value) for e in iteration_effectiveness]
    
    # Основной график
    plt.subplot(2, 1, 1)
    plt.plot(range(len(capped_effectiveness)), capped_effectiveness, 'o-', alpha=0.5)
    plt.axhline(y=1.0, color='r', linestyle='--', 
                label='Базовый уровень (случайное угадывание)')
    
    # Среднее значение эффективности
    mean_effectiveness = np.mean([e for e in iteration_effectiveness if e != float('inf')])
    plt.axhline(y=mean_effectiveness, color='g', linestyle='-', 
                label=f'Среднее значение: {mean_effectiveness:.2f}')
    
    plt.title('Эффективность ГСЧ по сравнению с теоретическими вероятностями')
    plt.xlabel('Номер итерации')
    plt.ylabel('Эффективность (1/теор.вероятность)')
    plt.ylim(0, max_display_value * 1.1)
    plt.legend()
    
    # Гистограмма эффективности
    plt.subplot(2, 1, 2)
    # Фильтруем бесконечные значения
    finite_effectiveness = [e for e in iteration_effectiveness if e != float('inf')]
    
    # Используем логарифмическую шкалу для лучшей визуализации
    plt.hist(finite_effectiveness, bins=50, alpha=0.7, color='green')
    plt.axvline(x=1.0, color='r', linestyle='--', 
                label='Базовый уровень (случайное угадывание)')
    plt.axvline(x=mean_effectiveness, color='b', linestyle='-', 
                label=f'Среднее значение: {mean_effectiveness:.2f}')
    
    plt.title('Распределение эффективности ГСЧ')
    plt.xlabel('Эффективность (1/теор.вероятность)')
    plt.ylabel('Частота')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('generator_effectiveness.png')
    plt.show()

if __name__ == "__main__":
    main()  # Укажите путь к вашему CSV файлу в качестве аргумента, если он отличается