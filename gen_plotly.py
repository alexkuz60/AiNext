import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.special import comb

# Параметры
max_num = 12  # Максимальное значение в диапазоне (от 1 до max_num)
input_arr_size = 500  # Размер входного массива

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
def main(file_path="2_12.txt"):
    start = time.time()
    # Загрузка данных
    data, n_columns = load_data_from_csv(file_path)
    
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
    
    print(time.time() - start)
    # Визуализация результатов с использованием Plotly
    
    # 1. График распределения совпадений
    fig1 = px.histogram(
        x=match_counts,
        nbins=n_columns + 1,
        range_x=[-0.5, n_columns + 0.5],
        labels={'x': 'Количество совпадений', 'y': 'Частота'},
        title='Распределение количества совпадений по строкам'
    )
    fig1.update_layout(bargap=0.1)
    fig1.write_html('distribution_of_matches.html')
    fig1.show()
    
    # 2. График благоприятности прогноза (совпадения по итерациям)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(len(match_counts))),
        y=match_counts,
        mode='markers+lines',
        name='Совпадения',
        marker=dict(size=6, opacity=0.6),
        line=dict(width=1)
    ))
    fig2.add_trace(go.Scatter(
        x=[0, len(match_counts) - 1],
        y=[np.mean(match_counts), np.mean(match_counts)],
        mode='lines',
        name=f'Среднее: {np.mean(match_counts):.2f}',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig2.update_layout(
        title='Благоприятность прогноза по итерациям',
        xaxis_title='Номер итерации',
        yaxis_title='Количество совпадений'
    )
    fig2.write_html('match_iterations.html')
    fig2.show()
    
    # 3. Тепловая карта матрицы переходов
    fig3 = px.imshow(
        transition_matrices[0],
        labels=dict(x="Следующее число", y="Текущее число", color="Вероятность"),
        x=[str(i+1) for i in range(max_num)],
        y=[str(i+1) for i in range(max_num)],
        text_auto='.2f',
        aspect="auto",
        title=f'Матрица переходов: столбец 1 -> столбец 2'
    )
    fig3.write_html('transition_matrix.html')
    fig3.show()
    
    # 4. Сравнение распределений по столбцам
    column_idx = 0  # Можно изменить для отображения других столбцов
    
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=data.iloc[:, column_idx].values,
        name=f'Оригинал (столбец {column_idx+1})',
        opacity=0.7,
        xbins=dict(start=0.5, end=max_num + 0.5, size=1),
        marker_color='blue'
    ))
    
    generated_column = [result['generated'][column_idx] for result in comparison_results]
    fig4.add_trace(go.Histogram(
        x=generated_column,
        name=f'Сгенерировано (столбец {column_idx+1})',
        opacity=0.5,
        xbins=dict(start=0.5, end=max_num + 0.5, size=1),
        marker_color='orange'
    ))
    
    fig4.update_layout(
        title=f'Сравнение распределений для столбца {column_idx+1}',
        xaxis_title='Значение',
        yaxis_title='Частота',
        barmode='overlay',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1)
    )
    fig4.write_html('column_distribution.html')
    fig4.show()
    
    # 5. Сравнение теоретических и фактических вероятностей
    fig5 = go.Figure()
    x = list(range(n_columns + 1))
    
    fig5.add_trace(go.Bar(
        x=x,
        y=theoretical_probs,
        name='Теоретическая',
        marker_color='blue',
        opacity=0.7
    ))
    
    fig5.add_trace(go.Bar(
        x=x,
        y=actual_probs,
        name='Фактическая',
        marker_color='orange',
        opacity=0.7
    ))
    
    fig5.update_layout(
        title='Сравнение теоретических и фактических вероятностей совпадений',
        xaxis_title='Количество совпадений',
        yaxis_title='Вероятность',
        barmode='group',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    fig5.write_html('probability_comparison.html')
    fig5.show()
    
    # 6. Отношение фактических вероятностей к теоретическим
    fig6 = go.Figure()
    
    # Ограничиваем очень большие значения для лучшей визуализации
    capped_ratio = [min(r, 10) if r != float('inf') else 10 for r in ratio_to_theoretical]
    
    fig6.add_trace(go.Bar(
        x=x,
        y=capped_ratio,
        marker_color='green',
        opacity=0.7
    ))
    
    fig6.add_trace(go.Scatter(
        x=[0, n_columns],
        y=[1, 1],
        mode='lines',
        name='Теоретическое ожидание',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig6.update_layout(
        title='Отношение фактических вероятностей к теоретическим',
        xaxis_title='Количество совпадений',
        yaxis_title='Отношение (факт/теория)',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    # Добавляем аннотации для очень больших значений
    for i, r in enumerate(ratio_to_theoretical):
        if r > 10 or r == float('inf'):
            fig6.add_annotation(
                x=i,
                y=10,
                text=f"∞" if r == float('inf') else f"{r:.1f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30
            )
    
    fig6.write_html('ratio_to_theoretical.html')
    fig6.show()
    
    # 7. Скользящее среднее совпадений
    window_size = min(50, len(match_counts) // 10)  # Размер окна для скользящего среднего
    rolling_mean = pd.Series(match_counts).rolling(window=window_size).mean()
    
    # Теоретическое ожидание среднего числа совпадений
    theoretical_mean = sum(i * theoretical_probs[i] for i in range(n_columns + 1))
    
    fig7 = go.Figure()
    
    fig7.add_trace(go.Scatter(
        x=list(range(len(match_counts))),
        y=match_counts,
        mode='markers',
        name='Совпадения',
        marker=dict(size=6, opacity=0.3, color='blue')
    ))
    
    fig7.add_trace(go.Scatter(
        x=list(range(len(rolling_mean))),
        y=rolling_mean,
        mode='lines',
        name=f'Скользящее среднее (окно={window_size})',
        line=dict(color='red', width=3)
    ))
    
    fig7.add_trace(go.Scatter(
        x=[0, len(match_counts) - 1],
        y=[theoretical_mean, theoretical_mean],
        mode='lines',
        name=f'Теоретическое среднее: {theoretical_mean:.2f}',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig7.update_layout(
        title='Динамика совпадений с течением времени',
        xaxis_title='Номер итерации',
        yaxis_title='Количество совпадений',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)')
    )
    fig7.write_html('match_dynamics.html')
    fig7.show()
    
    # 8. Эффективность ГСЧ по сравнению с теоретическими вероятностями
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
    
    # Создаем подграфики
    fig8 = make_subplots(rows=2, cols=1, 
                         subplot_titles=('Эффективность ГСЧ по сравнению с теоретическими вероятностями', 
                                         'Распределение эффективности ГСЧ'))
    
    # Основной график
    fig8.add_trace(
        go.Scatter(
            x=list(range(len(capped_effectiveness))),
            y=capped_effectiveness,
            mode='markers+lines',
            name='Эффективность',
            marker=dict(size=6, opacity=0.5),
            line=dict(width=1)
        ),
        row=1, col=1
    )
    
    fig8.add_trace(
        go.Scatter(
            x=[0, len(capped_effectiveness) - 1],
            y=[1, 1],
            mode='lines',
            name='Базовый уровень',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Среднее значение эффективности
    mean_effectiveness = np.mean([e for e in iteration_effectiveness if e != float('inf')])
    fig8.add_trace(
        go.Scatter(
            x=[0, len(capped_effectiveness) - 1],
            y=[mean_effectiveness, mean_effectiveness],
            mode='lines',
            name=f'Среднее: {mean_effectiveness:.2f}',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Гистограмма эффективности
    # Фильтруем бесконечные значения
    finite_effectiveness = [e for e in iteration_effectiveness if e != float('inf') and e <= max_display_value]
    
    fig8.add_trace(
        go.Histogram(
            x=finite_effectiveness,
            nbinsx=50,
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig8.add_trace(
        go.Scatter(
            x=[1, 1],
            y=[0, len(finite_effectiveness) // 5],  # Примерная высота для видимости линии
            mode='lines',
            name='Базовый уровень',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    fig8.add_trace(
        go.Scatter(
            x=[mean_effectiveness, mean_effectiveness],
            y=[0, len(finite_effectiveness) // 5],  # Примерная высота для видимости линии
            mode='lines',
            name=f'Среднее: {mean_effectiveness:.2f}',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig8.update_layout(
        height=800,
        showlegend=False,
        xaxis_title='Номер итерации',
        yaxis_title='Эффективность (1/теор.вероятность)',
        xaxis2_title='Эффективность (1/теор.вероятность)',
        yaxis2_title='Частота'
    )
    
    fig8.update_yaxes(range=[0, max_display_value * 1.1], row=1, col=1)
    
    fig8.write_html('generator_effectiveness.html')
    fig8.show()
    
    # 9. Комбинированный дашборд с основными метриками
    fig9 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Распределение совпадений',
            'Отношение к теоретическим вероятностям',
            'Динамика совпадений',
            'Эффективность генератора'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Распределение совпадений
    fig9.add_trace(
        go.Bar(
            x=list(range(n_columns + 1)),
            y=match_counts_hist,
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 2. Отношение к теоретическим вероятностям
    fig9.add_trace(
        go.Bar(
            x=list(range(n_columns + 1)),
            y=capped_ratio,
            marker_color='green',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig9.add_trace(
        go.Scatter(
            x=[0, n_columns],
            y=[1, 1],
            mode='lines',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=2
    )
    
    # 3. Динамика совпадений
    fig9.add_trace(
        go.Scatter(
            x=list(range(len(match_counts))),
            y=match_counts,
            mode='markers',
            marker=dict(size=4, opacity=0.3, color='blue')
        ),
        row=2, col=1
    )
    
    fig9.add_trace(
        go.Scatter(
            x=list(range(len(rolling_mean))),
            y=rolling_mean,
            mode='lines',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # 4. Эффективность генератора
    fig9.add_trace(
        go.Scatter(
            x=list(range(len(capped_effectiveness))),
            y=capped_effectiveness,
            mode='markers+lines',
            marker=dict(size=4, opacity=0.5),
            line=dict(width=1)
        ),
        row=2, col=2
    )
    
    fig9.add_trace(
        go.Scatter(
            x=[0, len(capped_effectiveness) - 1],
            y=[1, 1],
            mode='lines',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=2
    )
    
    # Обновляем оси и заголовки
    fig9.update_xaxes(title_text="Количество совпадений", row=1, col=1)
    fig9.update_yaxes(title_text="Частота", row=1, col=1)
    
    fig9.update_xaxes(title_text="Количество совпадений", row=1, col=2)
    fig9.update_yaxes(title_text="Отношение (факт/теория)", row=1, col=2)
    
    fig9.update_xaxes(title_text="Номер итерации", row=2, col=1)
    fig9.update_yaxes(title_text="Количество совпадений", row=2, col=1)
    
    fig9.update_xaxes(title_text="Номер итерации", row=2, col=2)
    fig9.update_yaxes(title_text="Эффективность", row=2, col=2)
    
    fig9.update_layout(
        height=800,
        title_text="Анализ генератора случайных чисел",
        showlegend=False
    )
    
    fig9.write_html('generator_dashboard.html')
    fig9.show()

if __name__ == "__main__":
    main()  # Укажите путь к вашему CSV файлу в качестве аргумента, если он отличается