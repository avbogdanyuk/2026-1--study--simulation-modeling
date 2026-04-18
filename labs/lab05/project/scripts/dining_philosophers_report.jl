# Формирование итогового отчёта по моделированию задачи "Обедающие философы"
#
# Данный скрипт загружает результаты моделирования классической сети
# и сети с арбитром, строит сравнительные графики состояния "Ест"
# для всех философов и сохраняет итоговый отчёт.

using DrWatson
@quickactivate "project"

using DataFrames, CSV, Plots

# Загрузка результатов моделирования
df_classic = CSV.read(datadir("dining_classic.csv"), DataFrame)
df_arbiter = CSV.read(datadir("dining_arbiter.csv"), DataFrame)

N = 5

# Столбцы для состояния "Ест"
eat_cols = [Symbol("Eat_$i") for i = 1:N]

# График для классической сети
p1 = plot(
    df_classic.time,
    Matrix(df_classic[:, eat_cols]),
    label = ["Ф $i" for i = 1:N],
    xlabel = "Время",
    ylabel = "Ест (1/0)",
    title = "Классическая сеть",
)

# График для сети с арбитром
p2 = plot(
    df_arbiter.time,
    Matrix(df_arbiter[:, eat_cols]),
    label = ["Ф $i" for i = 1:N],
    xlabel = "Время",
    ylabel = "Ест (1/0)",
    title = "Сеть с арбитром",
)

# Объединение графиков
p_final = plot(p1, p2, layout = (2, 1), size = (800, 600))

# Сохранение отчёта
savefig(plotsdir("final_report.png"))
println("Отчёт сохранён в plots/final_report.png")