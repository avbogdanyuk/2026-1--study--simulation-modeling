# # Модель SIR
# **Цель:** Исследовать динамику распространения инфекционного заболевания
#
# Модель SIR делит всю популяцию на три взаимосвязанные группы (компартменты), что отражено в её названии:
# — 𝑆 — Susceptible (Восприимчивые): люди, которые не болели, не имеют иммунитета и могут заразиться.
# — 𝐼 — Infectious (Инфицированные/Заразные): люди, которые в данный момент больны и могут передавать инфекцию.
# — 𝑅 — Recovered (Выздоровевшие/Удаленные): люди, которые переболели и приобрели иммунитет (или умерли). Они больше не участвуют в процессе передачи

# ## Инициализация проекта и загрузка пакетов
using DrWatson
@quickactivate "project"

using DifferentialEquations
using SimpleDiffEq
using Tables
using DataFrames
using StatsPlots
using LaTeXStrings # Для красивого отображения формул на графиках
using Plots
using BenchmarkTools
using CSV

# ## Определяем имя скрипта для организации выходных файлов
script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

# ## Определение модели
#
# Модель описывается системой дифференциальных уравнений:
# $$
# \begin{cases}
# \frac{dS}{dt} = -\beta \cdot c \cdot \frac{I}{N} \cdot S \\
# \frac{dI}{dt} = \beta \cdot c \cdot \frac{I}{N} \cdot S - \gamma \cdot I \\
# \frac{dR}{dt} = \gamma \cdot I
# \end{cases}
# $$
#
# где:
# *   \( \beta \) — вероятность передачи инфекции при контакте;
# *   \( c \) — среднее число контактов человека в единицу времени;
# *   \( \gamma \) — скорость выздоровления (\(1/\gamma\) — средняя продолжительность болезни).
function sir_ode!(du, u, p, t)
    (S, I, R) = u
    (β, c, γ) = p
    N = S + I + R # Общая численность популяции
    @inbounds begin
        du[1] = -β * c * I / N * S
        du[2] = β * c * I / N * S - γ * I
        du[3] = γ * I
    end
    nothing
end

# ## Функция для запуска одного эксперимента
#
# Эта функция принимает словарь параметров и возвращает словарь с результатами.
# Она будет использоваться с `produce_or_load` для автоматического кэширования.

function run_single_experiment(params::Dict)
    # Извлекаем параметры из словаря
    u0 = params[:u0]
    β = params[:β]
    c = params[:c]
    γ = params[:γ]
    tspan = params[:tspan]
    solver = params[:solver]
    saveat = params[:saveat]
    
    # Создаем и решаем задачу
    prob = ODEProblem(sir_ode!, u0, tspan, [β, c, γ])
    sol = solve(prob, solver; saveat=saveat)
    
    # Преобразуем решение в DataFrame для удобства
    df = DataFrame(Tables.table(sol'))
    rename!(df, ["S", "I", "R"])
    df[!, :t] = sol.t
    df[!, :N] = df.S + df.I + df.R
    
    # Анализ результатов
    peak_idx = argmax(df.I)
    peak_time = df.t[peak_idx]
    peak_value = df.I[peak_idx]
    
    final_S = last(df.S)
    final_I = last(df.I)
    final_R = last(df.R)
    
    # Расчет R0
    R0 = (c * β) / γ
    
    # Возвращаем словарь с результатами
    return Dict(
        "solution" => sol,
        "dataframe" => df,
        "peak_time" => peak_time,
        "peak_value" => peak_value,
        "final_S" => final_S,
        "final_I" => final_I,
        "final_R" => final_R,
        "R0" => R0,
        "parameters" => params
    )
end

# ## Базовый эксперимент с параметрами по умолчанию
#
# Зададим параметры для первого запуска.

base_params = Dict(
    :u0 => [990.0, 10.0, 0.0],     # [S0, I0, R0]
    :β => 0.05,                     # вероятность заражения при контакте
    :c => 10.0,                     # среднее число контактов
    :γ => 0.25,                     # скорость выздоровления
    :tspan => (0.0, 40.0),          # интервал времени
    :solver => Tsit5(),              # метод решения
    :saveat => 0.1,                  # шаг сохранения
    :experiment_name => "base_experiment"
)

println("Параметры базового эксперимента:")
for (key, value) in base_params
    println("  $key = $value")
end

# Запускаем базовый эксперимент с кэшированием результатов
println("\nЗапуск базового эксперимента...")
data_base, path_base = produce_or_load(
    datadir(script_name, "single"),  # Папка для сохранения
    base_params,                     # Параметры эксперимента
    run_single_experiment,           # Функция для выполнения
    prefix = "sir_base",             # Префикс имени файла
    tag = false,                      # Не добавлять git-тег
    verbose = true
)

println("\nРезультаты базового эксперимента:")
println("  R0 = ", round(data_base["R0"], digits=3))
println("  Пик эпидемии: I_max = ", round(data_base["peak_value"], digits=1),
        " на t = ", round(data_base["peak_time"], digits=1), " день")
println("  Итоговое число переболевших: R(∞) = ", round(data_base["final_R"], digits=1))
println("  Доля переболевших: ", round(data_base["final_R"]/sum(base_params[:u0])*100, digits=1), "%")
println("  Файл результатов: ", path_base)

# Сохраняем DataFrame базового эксперимента для дальнейшего использования
df_base = data_base["dataframe"]

# ## Визуализация базового эксперимента
#
# Построим основные графики для базового эксперимента.

# ## Основной график: динамика всех трех групп

plt1 = plot(df_base.t, [df_base.S, df_base.I, df_base.R],
            label=[L"S(t)" L"I(t)" L"R(t)"],
            xlabel="Время, дни", ylabel="Количество людей",
            title="Модель SIR: Динамика эпидемии (базовый эксперимент)",
            linewidth=2, legend=:right, grid=true, size=(800, 500))

# Добавляем аннотацию с параметрами
annotate!(plt1, 28, 800,
          text("Параметры:\nβ = $(base_params[:β])\nc = $(base_params[:c])\nγ = $(base_params[:γ])\nR0 = $(round(data_base["R0"], digits=2))", 10, :left))

savefig(plt1, plotsdir(script_name, "sir_base_main.png"))

# ## График только инфицированных (I)

plt2 = plot(df_base.t, df_base.I,
            label=L"I(t)", xlabel="Время, дни",
            ylabel="Количество инфицированных",
            title="Динамика числа зараженных (базовый эксперимент)",
            color=:red, linewidth=2, fill=(0, 0.3, :red),
            grid=true, size=(800, 400))

# Отмечаем пик эпидемии
vline!(plt2, [data_base["peak_time"]], color=:black, linestyle=:dash,
       linewidth=1, label=false)
annotate!(plt2, data_base["peak_time"], data_base["peak_value"] * 1.05,
          text("Пик: $(round(data_base["peak_value"], digits=1)) на $(round(data_base["peak_time"], digits=1)) день", 9, :top))

savefig(plt2, plotsdir(script_name, "sir_base_infected.png"))

# ## График эффективного репродуктивного числа Re

df_base[!, :Re] = data_base["R0"] .* df_base.S ./ df_base.N

plt3 = plot(df_base.t, df_base.Re,
            label=L"R_e(t) = R_0 \cdot S(t)/N",
            xlabel="Время, дни", ylabel="R_e",
            title="Эффективное репродуктивное число",
            color=:green, linewidth=2, grid=true, size=(800, 400))

# Отмечаем критический уровень Re = 1
hline!(plt3, [1.0], color=:red, linestyle=:dash,
       linewidth=1.5, label="Re = 1 (порог затухания)")

savefig(plt3, plotsdir(script_name, "sir_base_effective_R.png"))

println("\nГрафики базового эксперимента сохранены в: ", plotsdir(script_name))

# ## Параметрическое исследование
#
# Исследуем влияние ключевых параметров на динамику эпидемии.
# Будем варьировать:
# * Коэффициент заражения β (вероятность передачи при контакте)
# * Число контактов c (социальная активность)
# * Скорость выздоровления γ (качество медицинской помощи)

println("\n" * "="^60)
println("ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ")
println("="^60)

# ## Сетка параметров для сканирования
#
# Определяем наборы значений для каждого параметра.

param_grid = Dict(
    :u0 => [[990.0, 10.0, 0.0]],                    # фиксируем начальные условия
    :β => [0.01, 0.03, 0.05, 0.07, 0.09],           # варьируем вероятность заражения
    :c => [5.0, 10.0, 15.0, 20.0],                  # варьируем число контактов
    :γ => [0.1, 0.2, 0.25, 0.33, 0.5],              # варьируем скорость выздоровления
    :tspan => [(0.0, 60.0)],                         # увеличиваем время для медленных сценариев
    :solver => [Tsit5()],                            # фиксируем метод решения
    :saveat => [0.1],                                 # фиксируем шаг сохранения
    :experiment_name => ["parametric_scan"]
)

# Генерируем все комбинации параметров
all_params = dict_list(param_grid)

println("Сетка параметров:")
println("  β (вероятность заражения): ", param_grid[:β])
println("  c (число контактов): ", param_grid[:c])
println("  γ (скорость выздоровления): ", param_grid[:γ])
println("Всего комбинаций: ", length(all_params))

# ## Запуск всех экспериментов и сбор результатов
#
# Запускаем симуляции для всех комбинаций параметров.

all_results = []  # для хранения сводных результатов
all_dfs = []      # для хранения полных данных

for (i, params) in enumerate(all_params)
    # Запускаем или загружаем из кэша
    data, path = produce_or_load(
        datadir(script_name, "parametric_scan"),
        params,
        run_single_experiment,
        prefix = "sir_scan",
        tag = false,
        verbose = false
    )
    
    # Сохраняем сводные результаты
    result_summary = merge(
        params,
        Dict(
            :R0 => data["R0"],
            :peak_time => data["peak_time"],
            :peak_value => data["peak_value"],
            :final_R => data["final_R"],
            :final_percent => data["final_R"] / sum(params[:u0]) * 100,
            :filepath => path
        )
    )
    push!(all_results, result_summary)
    
    # Сохраняем полные данные для визуализации (только для некоторых, чтобы не перегружать память)
    if params[:β] in [0.03, 0.05, 0.07] && params[:c] == 10.0 && params[:γ] == 0.25
        push!(all_dfs, (params=params, df=data["dataframe"]))
    end
end

# Преобразуем результаты в DataFrame для анализа
results_df = DataFrame(all_results)

println("\nПервые 5 строк сводной таблицы результатов:")
println(first(results_df, 5))

# Сохраняем сводные результаты в CSV
CSV.write(datadir(script_name, "parametric_results.csv"), results_df)
println("\nСводные результаты сохранены: ", datadir(script_name, "parametric_results.csv"))

# ## Анализ влияния параметров на R0
#
# R0 = (c * β) / γ — теоретическая зависимость.

println("\n" * "-"^60)
println("Анализ влияния параметров на R0")
println("-"^60)

# Группировка по β при фиксированных c и γ
println("\nВлияние β (при c=10, γ=0.25):")
subset_df = filter(row -> row.c == 10.0 && row.γ == 0.25, results_df)
sort!(subset_df, :β)
for row in eachrow(subset_df)
    println("  β = $(row.β): R0 = $(round(row.R0, digits=2)), пик I = $(round(row.peak_value, digits=1)), переболело $(round(row.final_percent, digits=1))%")
end

# Группировка по c при фиксированных β и γ
println("\nВлияние c (при β=0.05, γ=0.25):")
subset_df = filter(row -> row.β == 0.05 && row.γ == 0.25, results_df)
sort!(subset_df, :c)
for row in eachrow(subset_df)
    println("  c = $(row.c): R0 = $(round(row.R0, digits=2)), пик I = $(round(row.peak_value, digits=1)), переболело $(round(row.final_percent, digits=1))%")
end

# Группировка по γ при фиксированных β и c
println("\nВлияние γ (при β=0.05, c=10):")
subset_df = filter(row -> row.β == 0.05 && row.c == 10.0, results_df)
sort!(subset_df, :γ)
for row in eachrow(subset_df)
    println("  γ = $(row.γ) (длительность болезни = $(round(1/row.γ, digits=1)) дн): R0 = $(round(row.R0, digits=2)), пик I = $(round(row.peak_value, digits=1)), переболело $(round(row.final_percent, digits=1))%")
end

# ## Визуализация параметрического исследования
#
# Создадим серию графиков для анализа влияния параметров.

# ## Сравнение динамики I(t) для разных β

println("\nПостроение графиков параметрического исследования...")

# Фильтруем данные для фиксированных c и γ
plot_data = filter(row -> row.c == 10.0 && row.γ == 0.25, all_dfs)

if !isempty(plot_data)
    plt4 = plot(xlabel="Время, дни", ylabel="Количество инфицированных",
                title="Влияние β на динамику эпидемии (c=10, γ=0.25)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))
    
    for item in plot_data
        β_val = item.params[:β]
        df = item.df
        plot!(plt4, df.t, df.I, label=L"β = %$β_val")
    end
    
    savefig(plt4, plotsdir(script_name, "param_beta_comparison.png"))
    println("  Сохранен график: param_beta_comparison.png")
end

# ## Сравнение динамики I(t) для разных c

plot_data = filter(row -> row.β == 0.05 && row.γ == 0.25, all_dfs)

if !isempty(plot_data)
    plt5 = plot(xlabel="Время, дни", ylabel="Количество инфицированных",
                title="Влияние числа контактов c на динамику (β=0.05, γ=0.25)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))
    
    for item in plot_data
        c_val = item.params[:c]
        df = item.df
        plot!(plt5, df.t, df.I, label=L"c = %$c_val")
    end
    
    savefig(plt5, plotsdir(script_name, "param_c_comparison.png"))
    println("  Сохранен график: param_c_comparison.png")
end

# ## Сравнение динамики I(t) для разных γ

plot_data = filter(row -> row.β == 0.05 && row.c == 10.0, all_dfs)

if !isempty(plot_data)
    plt6 = plot(xlabel="Время, дни", ylabel="Количество инфицированных",
                title="Влияние скорости выздоровления γ на динамику (β=0.05, c=10)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))
    
    for item in plot_data
        γ_val = item.params[:γ]
        df = item.df
        plot!(plt6, df.t, df.I, label=L"γ = %$γ_val (1/γ = $(round(1/γ_val, digits=1)) дн)")
    end
    
    savefig(plt6, plotsdir(script_name, "param_gamma_comparison.png"))
    println("  Сохранен график: param_gamma_comparison.png")
end

# ## Зависимость пикового значения I_max от R0

plt7 = scatter(results_df.R0, results_df.peak_value,
               xlabel=L"R_0 = (c \cdot \beta)/\gamma",
               ylabel="Пиковое число зараженных I_max",
               title="Зависимость пика эпидемии от R0",
               markersize=6, markercolor=:red, alpha=0.7,
               grid=true, size=(800, 500), legend=false)

# Добавляем аппроксимацию
sort!(results_df, :R0)
plot!(plt7, results_df.R0, results_df.peak_value,
      linewidth=2, color=:blue, linestyle=:dash, label="тренд")

savefig(plt7, plotsdir(script_name, "param_peak_vs_R0.png"))
println("  Сохранен график: param_peak_vs_R0.png")

# ## Зависимость итогового числа переболевших от R0

plt8 = scatter(results_df.R0, results_df.final_percent,
               xlabel=L"R_0 = (c \cdot \beta)/\gamma",
               ylabel="Итоговая доля переболевших, %",
               title="Зависимость итогового размера эпидемии от R0",
               markersize=6, markercolor=:green, alpha=0.7,
               grid=true, size=(800, 500), legend=false)

# Добавляем теоретическую кривую для сравнения
R0_range = 0.5:0.1:4.0
theoretical_final_percent = 100 .* (1 .- 1 ./ R0_range)
plot!(plt8, R0_range, theoretical_final_percent,
      linewidth=2, color=:black, linestyle=:dash,
      label="Теория: 1 - 1/R0")

savefig(plt8, plotsdir(script_name, "param_final_vs_R0.png"))
println("  Сохранен график: param_final_vs_R0.png")

# ## Тепловая карта: зависимость I_max от β и c при фиксированном γ

using StatsPlots

# Подготовка данных для тепловой карты
γ_fixed = 0.25
heatmap_data = filter(row -> row.γ == γ_fixed, results_df)

if nrow(heatmap_data) > 0
    β_unique = sort(unique(heatmap_data.β))
    c_unique = sort(unique(heatmap_data.c))
    
    # Создаем матрицу для тепловой карты
    peak_matrix = zeros(length(β_unique), length(c_unique))
    for (i, β_val) in enumerate(β_unique)
        for (j, c_val) in enumerate(c_unique)
            row = filter(row -> row.β == β_val && row.c == c_val, heatmap_data)
            if nrow(row) > 0
                peak_matrix[i, j] = row.peak_value[1]
            end
        end
    end
    
    plt9 = heatmap(c_unique, β_unique, peak_matrix,
                   xlabel="Число контактов c", ylabel="Вероятность заражения β",
                   title="Тепловая карта: пик заражений I_max (γ=0.25)",
                   color=:viridis, colorbar_title="I_max",
                   size=(800, 600))
    
    savefig(plt9, plotsdir(script_name, "param_heatmap.png"))
    println("  Сохранен график: param_heatmap.png")
end

# ## Анализ пороговых значений
#
# Найдем комбинации параметров, при которых эпидемия затухает (R0 < 1).

println("\n" * "-"^60)
println("АНАЛИЗ ПОРОГОВЫХ ЗНАЧЕНИЙ")
println("-"^60)

# Сценарии с затухающей эпидемией
extinct_scenarios = filter(row -> row.R0 < 1.0, results_df)

if nrow(extinct_scenarios) > 0
    println("\nСценарии с затухающей эпидемией (R0 < 1):")
    for row in eachrow(extinct_scenarios)
        println("  β=$(row.β), c=$(row.c), γ=$(row.γ) -> R0=$(round(row.R0, digits=3))")
    end
else
    println("\nНет сценариев с затухающей эпидемией в текущей сетке параметров.")
end

# Критическое значение c для фиксированных β и γ
β_fixed = 0.05
γ_fixed = 0.25
c_critical = γ_fixed / β_fixed

println("\nКритический анализ для β=0.05, γ=0.25:")
println("  Критическое число контактов (R0=1): c_critical = γ/β = ", round(c_critical, digits=2))
println("  При c < $(round(c_critical, digits=2)) эпидемия затухает, при c > растет.")

# ## Бенчмаркинг производительности для разных параметров
#
# Оценим, как меняется время вычисления в зависимости от параметров.

println("\n" * "="^60)
println("БЕНЧМАРКИНГ ПРОИЗВОДИТЕЛЬНОСТИ")
println("="^60)

benchmark_results = []

for β_val in [0.01, 0.05, 0.09]
    bench_params = Dict(
        :u0 => [990.0, 10.0, 0.0],
        :β => β_val,
        :c => 10.0,
        :γ => 0.25,
        :tspan => (0.0, 60.0),
        :solver => Tsit5(),
        :saveat => 0.1
    )
    
    function bench_run()
        prob = ODEProblem(sir_ode!, bench_params[:u0], bench_params[:tspan], 
                         [bench_params[:β], bench_params[:c], bench_params[:γ]])
        return solve(prob, bench_params[:solver]; saveat=bench_params[:saveat])
    end
    
    println("\nБенчмарк для β = $β_val:")
    b = @benchmark $bench_run() samples=50 evals=1
    time_ms = median(b).time / 1e6  # время в миллисекундах
    push!(benchmark_results, (β=β_val, time_ms=time_ms))
    println("  Среднее время: ", round(time_ms, digits=2), " мс")
end

bench_df = DataFrame(benchmark_results)

# ## Сохранение всех результатов
#
# Сохраняем все данные для последующего анализа.

@save datadir(script_name, "all_results.jld2") 
    base_params param_grid all_params results_df bench_df df_base

@save datadir(script_name, "all_plots.jld2") 
    plt1 plt2 plt3 plt4 plt5 plt6 plt7 plt8 plt9



