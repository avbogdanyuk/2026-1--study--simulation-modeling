using DrWatson                      # Управление научными проектами
@quickactivate "project"            # Активация окружения проекта

include(srcdir("DiningPhilosophers.jl"))

using .DiningPhilosophers           # Использование функций модуля
using DataFrames, CSV, Plots        # Работа с данными и визуализация

N = 5                               # Количество философов
tmax = 50.0                         # Максимальное время моделирования

println("="^60)
println("МОДЕЛИРОВАНИЕ ЗАДАЧИ 'ОБЕДАЮЩИЕ ФИЛОСОФЫ'")
println("="^60)
println("Количество философов: $N")
println("Максимальное время: $tmax")
println()

println("=== 1. Классическая сеть (без арбитра) ===")

net_classic, u0_classic, _ = build_classical_network(N)

df_classic = simulate_stochastic(net_classic, u0_classic, tmax)

CSV.write(datadir("dining_classic.csv"), df_classic)
println("Результаты сохранены: data/dining_classic.csv")

dead = detect_deadlock(df_classic, net_classic)
println("Deadlock обнаружен: $dead")

plot_classic = plot_marking_evolution(df_classic, N)

savefig(plotsdir("classic_simulation.png"))
println("График сохранён: plots/classic_simulation.png")
println()

println("=== 2. Сеть с арбитром ===")

net_arb, u0_arb, _ = build_arbiter_network(N)

df_arb = simulate_stochastic(net_arb, u0_arb, tmax)

CSV.write(datadir("dining_arbiter.csv"), df_arb)
println("Результаты сохранены: data/dining_arbiter.csv")

dead_arb = detect_deadlock(df_arb, net_arb)
println("Deadlock обнаружен: $dead_arb")

plot_arb = plot_marking_evolution(df_arb, N)

savefig(plotsdir("arbiter_simulation.png"))
println("График сохранён: plots/arbiter_simulation.png")
