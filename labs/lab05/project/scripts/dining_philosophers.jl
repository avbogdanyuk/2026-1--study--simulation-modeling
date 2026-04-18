# Сравнительное моделирование задачи "Обедающие философы"
# 
# Данный скрипт выполняет сравнительное моделирование классической сети Петри
# и сети с арбитром для задачи "Обедающие философы". Результаты сохраняются
# в виде CSV-файлов и графиков.

# Инициализация проекта и загрузка зависимостей

using DrWatson                      # Управление научными проектами
@quickactivate "project"            # Активация окружения проекта

# Подключение модуля DiningPhilosophers
include(srcdir("DiningPhilosophers.jl"))

using .DiningPhilosophers           # Использование функций модуля
using DataFrames, CSV, Plots        # Работа с данными и визуализация

# Настройка параметров моделирования

N = 5                               # Количество философов
tmax = 50.0                         # Максимальное время моделирования

println("="^60)
println("МОДЕЛИРОВАНИЕ ЗАДАЧИ 'ОБЕДАЮЩИЕ ФИЛОСОФЫ'")
println("="^60)
println("Количество философов: $N")
println("Максимальное время: $tmax")
println()

# 1. Моделирование классической сети (без арбитра)

println("=== 1. Классическая сеть (без арбитра) ===")

# Построение классической сети Петри
net_classic, u0_classic, _ = build_classical_network(N)

# Запуск стохастического моделирования
df_classic = simulate_stochastic(net_classic, u0_classic, tmax)

# Сохранение результатов в CSV
CSV.write(datadir("dining_classic.csv"), df_classic)
println("Результаты сохранены: data/dining_classic.csv")

# Обнаружение deadlock
dead = detect_deadlock(df_classic, net_classic)
println("Deadlock обнаружен: $dead")

# Визуализация эволюции маркировки
plot_classic = plot_marking_evolution(df_classic, N)

# Сохранение графика
savefig(plotsdir("classic_simulation.png"))
println("График сохранён: plots/classic_simulation.png")
println()

# 2. Моделирование сети с арбитром

println("=== 2. Сеть с арбитром ===")

# Построение сети Петри с арбитром
net_arb, u0_arb, _ = build_arbiter_network(N)

# Запуск стохастического моделирования
df_arb = simulate_stochastic(net_arb, u0_arb, tmax)

# Сохранение результатов в CSV
CSV.write(datadir("dining_arbiter.csv"), df_arb)
println("Результаты сохранены: data/dining_arbiter.csv")

# Обнаружение deadlock
dead_arb = detect_deadlock(df_arb, net_arb)
println("Deadlock обнаружен: $dead_arb")

# Визуализация эволюции маркировки
plot_arb = plot_marking_evolution(df_arb, N)

# Сохранение графика
savefig(plotsdir("arbiter_simulation.png"))
println("График сохранён: plots/arbiter_simulation.png")