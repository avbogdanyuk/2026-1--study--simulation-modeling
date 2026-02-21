# # Параметрическое исследование экспоненциального роста
#
# ## Активация проекта и загрузка пакетов
#
# **ИЗМЕНЕНИЕ:** Добавлен DrWatson для управления проектом и параметрами
using DrWatson
@quickactivate "project" # Активация проекта DrWatson
using DifferentialEquations
using DataFrames
using Plots
using JLD2
using BenchmarkTools
# Установка каталогов
script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))
# ## Определение модели
# Модель: 𝑑𝑢/𝑑𝑡 = 𝛼 ⋅ 𝑢
function exponential_growth!(du, u, p, t)
α = p.α # **ИЗМЕНЕНИЕ:** Параметры теперь передаются как именованный кортеж
du[1] = α * u[1]
end
# ## Определение параметров в Dict
# **ОСНОВНОЕ ИЗМЕНЕНИЕ:** Все параметры собраны в Dict для систематизации
# Базовый набор параметров (один эксперимент)
base_params = Dict(
:u0 => [1.0], # начальная популяция
:α => 0.3, # скорость роста
:tspan => (0.0, 10.0), # интервал времени
:solver => Tsit5(), # метод решения
:saveat => 0.1, # шаг сохранения результатов
:experiment_name => "base_experiment"
)
println("Базовые параметры эксперимента:")
for (key, value) in base_params
println(" $key = $value")
end
# ## Функция-обертка для запуска одного эксперимента
# **ИСПРАВЛЕНИЕ:** Возвращаем Dict со строковыми ключами
function run_single_experiment(params::Dict)
@unpack u0, α, tspan, solver, saveat = params
prob = ODEProblem(exponential_growth!, u0, tspan, (α=α,)) 
#Создаем и решаем задачу
sol = solve(prob, solver; saveat=saveat)
final_population = last(sol.u)[1] # Анализ результатов
doubling_time = log(2) / α
return Dict(
"solution" => sol,
"time_points" => sol.t,
"population_values" => first.(sol.u),
"final_population" => final_population,
"doubling_time" => doubling_time,
"parameters" => params # Сохраняем исходные параметры
) # Используем строки как ключи для совместимости с DrWatson
end
# ## Запуск базового эксперимента
# **ИЗМЕНЕНИЕ:** Используем produce_or_load для автоматического кэширования
data, path = produce_or_load(
datadir(script_name, "single"), # Папка для сохранения
base_params, # Параметры эксперимента
run_single_experiment, # Функция для выполнения
prefix = "exp_growth", # Префикс имени файла
tag = false, # Не добавлять git-тег
verbose = true
)
println("\nРезультаты базового эксперимента:")
println(" Финальная популяция: ", data["final_population"])
println(" Время удвоения: ", round(data["doubling_time"]; digits=2))
println(" Файл результатов: ", path)
# ## Визуализация базового эксперимента
p1 = plot(data["time_points"], data["population_values"],
label="α = $(base_params[:α])",
xlabel="Время, t",
ylabel="Популяция, u(t)",
title="Экспоненциальный рост (базовый эксперимент)",
lw=2,
legend=:topleft,
grid=true
)
# Сохраним график в папку plots
savefig(plotsdir(script_name, "single_experiment.png"))
# ## Параметрическое сканирование
# **НОВАЯ СЕКЦИЯ:** Исследование влияния параметра α
# Сетка параметров для сканирования
param_grid = Dict(
:u0 => [[1.0]], # фиксируем начальное условие
:α => [0.1, 0.3, 0.5, 0.8, 1.0], # исследуемые значения скорости роста
:tspan => [(0.0, 10.0)], # фиксируем интервал времени
:solver => [Tsit5()], # фиксируем метод решения
:saveat => [0.1], # фиксируем шаг сохранения
:experiment_name => ["parametric_scan"]
)
# Генерация всех комбинаций параметров
all_params = dict_list(param_grid)
println("\n" * "="^60)
println("ПАРАМЕТРИЧЕСКОЕ СКАНИРОВАНИЕ")
println("Всего комбинаций параметров: ", length(all_params))
println("Исследуемые значения α: ", param_grid[:α])
println("="^60)
# ## Запуск всех экспериментов и собор результатов
# **НОВАЯ СЕКЦИЯ:** Автоматический запуск и сохранение всех вариантов
all_results = []
all_dfs = []
for (i, params) in enumerate(all_params)
println("Прогресс: $i/$(length(all_params)) | α = $(params[:α])")
data, path = produce_or_load(
datadir(script_name, "parametric_scan"), # Данные
params, # Текущий набор параметров
run_single_experiment, # Функция для выполнения
prefix = "scan", # Префикс имени файла
tag = false,
verbose = false # Не выводить подробности для каждого запуска
) # Автоматическое сохранение/загрузка каждого эксперимента
result_summary = merge(
params,
Dict(
:final_population => data["final_population"],
:doubling_time => data["doubling_time"],
:filepath => path # Путь к сохраненным данным
)
) # Сохраняем сводные результаты (используем символы для параметров, но данные из data - строки)
push!(all_results, result_summary)
df = DataFrame(
t = data["time_points"],
u = data["population_values"],
α = fill(params[:α], length(data["time_points"]))
) # Сохраняем полные данные для визуализации
push!(all_dfs, df)
end
# ## Анализ и визуализация результатов сканирования
# **НОВАЯ СЕКЦИЯ:** Сравнительный анализ всех экспериментов
# Сводная таблица результатов
results_df = DataFrame(all_results)
println("\nСводная таблица результатов:")
println(results_df[!, [:α, :final_population, :doubling_time]])
# Сравнительный график всех траекторий
p2 = plot(size=(800, 500), dpi=150)
for params in all_params
data, _ = produce_or_load(
datadir(script_name, "parametric_scan"),
params,
run_single_experiment,
prefix = "scan"
) # Загружаем данные (они уже есть на диске)
plot!(p2, data["time_points"], data["population_values"],
label="α = $(params[:α])",
lw=2,
alpha=0.8
)
end
plot!(p2,
xlabel="Время, t",
ylabel="Популяция, u(t)",
title="Параметрическое исследование: влияние α на рост",
legend=:topleft,
grid=true
)
# Сохраним график в папку plots
savefig(plotsdir(script_name, "parametric_scan_comparison.png"))
# График зависимости времени удвоения от α
p3 = plot(results_df.α, results_df.doubling_time,
seriestype=:scatter,
label="Численное решение",
xlabel="Скорость роста, α",
ylabel="Время удвоения, t₂",
title="Зависимость времени удвоения от α",
markersize=8,
markercolor=:red,
legend=:topright
)
# Теоретическая кривая: t₂ = ln(2)/α
α_range = 0.1:0.01:1.0
plot!(p3, α_range, log(2) ./ α_range,
label="Теория: t₂ = ln(2)/α",
lw=2,
linestyle=:dash,
linecolor=:blue
)
# Сохраним график в папку plots
savefig(plotsdir(script_name, "doubling_time_vs_alpha.png"))
# ## Бенчмаркинг с разными параметрами
# **ИЗМЕНЕНИЕ:** Бенчмаркинг для разных значений α
println("\n" * "="^60)
println("Бенчмаркинг для разных значений α")
println("="^60)
benchmark_results = []
for α_value in param_grid[:α]
bench_params = Dict(
:u0 => [1.0],
:α => α_value,
:tspan => (0.0, 10.0),
:solver => Tsit5(),
:saveat => 0.1
) # Подготавливаем параметры для бенчмарка
function benchmark_run() # Функция для бенчмарка
prob = ODEProblem(exponential_growth!,
bench_params[:u0],
bench_params[:tspan],
(α=bench_params[:α],))
return solve(prob, bench_params[:solver];
saveat=bench_params[:saveat])
end
println("\nБенчмарк для α = $α_value:")
b = @benchmark $benchmark_run() samples=100 evals=1 # Запуск бенчмарка
push!(benchmark_results, (α=α_value, time=median(b).time/1e9)) # время в секундах
println(" Среднее время: ", round(median(b).time/1e9;digits=4), " сек")
end
# График зависимости времени вычисления от α
bench_df = DataFrame(benchmark_results)
p4 = plot(bench_df.α, bench_df.time,
seriestype=:scatter,
label="Время вычисления",
xlabel="Скорость роста, α",
ylabel="Время вычисления, сек",
title="Зависимость времени вычисления от α",
markersize=8,
markercolor=:green,
legend=:topleft
)
# Сохраним график в папку plots
savefig(plotsdir(script_name, "computation_time_vs_alpha.png"))
# ## Сохранение всех результатов
# **НОВАЯ СЕКЦИЯ:** Сохранение сводных данных для последующего
анализа
@save datadir(script_name, "all_results.jld2") base_params
param_grid all_params results_df bench_df
@save datadir(script_name, "all_plots.jld2") p1 p2 p3 p4
println("\n" * "="^60)
println("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
println("="^60)
println("\nРезультаты сохранены в:")
println(" • data/$(script_name)/single/ - базовый эксперимент")
println(" • data/$(script_name)/parametric_scan/ -параметрическое сканирование")
println(" • data/$(script_name)/all_results.jld2 - сводныеданные")
println(" • plots/$(script_name)/ - все графики")
println(" • data/$(script_name)/all_plots.jld2 - объектыграфиков")
println("\nДля анализа результатов используйте:")
println(" using JLD2, DataFrames")
println(" @load \"data/$(script_name)/all_results.jld2\"")
println(" println(results_df)")
