using DrWatson
@quickactivate "project"
using DifferentialEquations
using DataFrames
using StatsPlots
using LaTeXStrings
using Plots
using Statistics
using FFTW
script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

function lotka_volterra!(du, u, p, t)
x, y = u # x - жертвы, y - хищники
α, β, δ, γ = p # параметры модели
@inbounds begin
du[1] = α*x - β*x*y # уравнение для жертв
du[2] = δ*x*y - γ*y # уравнение для хищников
end
nothing
end

function run_single_experiment(params::Dict)

    u0 = params[:u0]
    α = params[:α]
    β = params[:β]
    δ = params[:δ]
    γ = params[:γ]
    tspan = params[:tspan]
    solver = params[:solver]
    saveat = params[:saveat]

    prob = ODEProblem(lotka_volterra!, u0, tspan, [α, β, δ, γ])
    sol = solve(prob, solver; saveat=saveat, reltol=1e-8, abstol=1e-10)

    df = DataFrame()
    df[!, :t] = sol.t
    df[!, :prey] = [u[1] for u in sol.u]
    df[!, :predator] = [u[2] for u in sol.u]

    x_star = γ / δ
    y_star = α / β

    peak_time_prey, peak_value_prey = find_first_peak(df.prey, df.t)
    peak_time_predator, peak_value_predator = find_first_peak(df.predator, df.t)

    phase_shift = if !isnan(peak_time_prey) && !isnan(peak_time_predator)
        peak_time_predator - peak_time_prey
    else
        NaN
    end

    freq_prey, spectrum_prey = compute_fft(df.prey .- mean(df.prey), saveat)
    if length(spectrum_prey) > 2
        idx_prey = argmax(spectrum_prey[2:end]) + 1
        dominant_period = 1 / freq_prey[idx_prey]
    else
        dominant_period = NaN
    end

    mean_prey = mean(df.prey)
    mean_predator = mean(df.predator)
    std_prey = std(df.prey)
    std_predator = std(df.predator)

    theoretical_period = 2π / sqrt(α * γ)

    return Dict(
        "solution" => sol,
        "dataframe" => df,
        "x_star" => x_star,
        "y_star" => y_star,
        "peak_time_prey" => peak_time_prey,
        "peak_value_prey" => peak_value_prey,
        "peak_time_predator" => peak_time_predator,
        "peak_value_predator" => peak_value_predator,
        "phase_shift" => phase_shift,
        "dominant_period" => dominant_period,
        "theoretical_period" => theoretical_period,
        "mean_prey" => mean_prey,
        "mean_predator" => mean_predator,
        "std_prey" => std_prey,
        "std_predator" => std_predator,
        "parameters" => params
    )
end

"""
    find_first_peak(signal, time)

Находит первый локальный максимум в сигнале и соответствующее время.
Возвращает (время_пика, значение_пика) или (NaN, NaN) если пик не найден.
"""
function find_first_peak(signal, time)
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            return time[i], signal[i]
        end
    end
    return NaN, NaN
end

"""
    compute_fft(signal, dt)

Вычисляет амплитудный спектр вещественного сигнала с помощью БПФ.
Возвращает частоты и соответствующие амплитуды.
"""
function compute_fft(signal, dt)
    n = length(signal)
    spectrum = abs.(rfft(signal))           # амплитудный спектр
    freq = rfftfreq(n, 1/dt)                # соответствующие частоты
    return freq, spectrum
end

println("\n" * "="^60)
println("БАЗОВЫЙ ЭКСПЕРИМЕНТ")
println("="^60)

base_params = Dict(
    :u0 => [40.0, 9.0],           # [x₀, y₀] начальная популяция
    :α => 0.1,                     # скорость размножения жертв
    :β => 0.02,                    # скорость поедания жертв
    :δ => 0.01,                    # коэффициент конверсии
    :γ => 0.3,                     # смертность хищников
    :tspan => (0.0, 200.0),        # интервал времени
    :solver => Tsit5(),             # метод решения
    :saveat => 0.1,                 # шаг сохранения
    :experiment_name => "base_experiment"
)

println("Параметры базового эксперимента:")
for (key, value) in base_params
    println("  $key = $value")
end

x_star_base = base_params[:γ] / base_params[:δ]
y_star_base = base_params[:α] / base_params[:β]
println("\nСтационарные точки:")
println("  x* = γ/δ = ", round(x_star_base, digits=2))
println("  y* = α/β = ", round(y_star_base, digits=2))

println("\nЗапуск базового эксперимента...")
data_base, path_base = produce_or_load(
    datadir(script_name, "single"),  # Папка для сохранения
    base_params,                     # Параметры эксперимента
    run_single_experiment,           # Функция для выполнения
    prefix = "lv_base",               # Префикс имени файла
    tag = false,                      # Не добавлять git-тег
    verbose = true
)

println("\nРезультаты базового эксперимента:")
println("  Средняя численность жертв: ", round(data_base["mean_prey"], digits=2))
println("  Средняя численность хищников: ", round(data_base["mean_predator"], digits=2))
println("  Период колебаний (Фурье): ", round(data_base["dominant_period"], digits=2), " ед. времени")
println("  Теоретический период: T ≈ 2π/√(αγ) = ", round(data_base["theoretical_period"], digits=2))
println("  Сдвиг фаз: ", round(data_base["phase_shift"], digits=2), " ед. времени")
println("  Файл результатов: ", path_base)

df_base = data_base["dataframe"]

plt1 = plot(df_base.t, [df_base.prey df_base.predator],
            label=[L"Жертвы (x)" L"Хищники (y)"],
            xlabel="Время", ylabel="Популяция",
            title="Модель Лотки-Вольтерры: Динамика популяций (базовый эксперимент)",
            linewidth=2, legend=:topright, grid=true,
            size=(900, 500), color=[:green :red])

hline!(plt1, [x_star_base], color=:green, linestyle=:dash, alpha=0.5,
       label=L"x^* (равновесие жертв)")
hline!(plt1, [y_star_base], color=:red, linestyle=:dash, alpha=0.5,
       label=L"y^* (равновесие хищников)")

savefig(plt1, plotsdir(script_name, "lv_base_dynamics.png"))

plt2 = plot(df_base.prey, df_base.predator,
            label="Фазовая траектория",
            xlabel="Популяция жертв (x)",
            ylabel="Популяция хищников (y)",
            title="Фазовый портрет (базовый эксперимент)",
            color=:blue, linewidth=1.5, grid=true,
            size=(800, 600), legend=:topright)

step = 50
for i in 1:step:length(df_base.prey)-step
    plot!(plt2, [df_base.prey[i], df_base.prey[i+step]],
                 [df_base.predator[i], df_base.predator[i+step]],
          arrow=:closed, color=:blue, alpha=0.3, label=false)
end

scatter!(plt2, [x_star_base], [y_star_base],
         color=:black, markersize=8,
         label=L"(x^*, y^*)")

x_range = LinRange(0, maximum(df_base.prey)*1.1, 100)
y_nulcline = base_params[:α] ./ (base_params[:β] .* x_range)
plot!(plt2, x_range, y_nulcline,
      color=:red, linestyle=:dash, linewidth=1.5,
      label="Изоклина хищников (dy/dt = 0)")
vline!(plt2, [base_params[:γ] / base_params[:δ]],
       color=:green, linestyle=:dash, linewidth=1.5,
       label="Изоклина жертв (dx/dt = 0)")

savefig(plt2, plotsdir(script_name, "lv_base_phase_portrait.png"))

println("\nГрафики базового эксперимента сохранены в: ", plotsdir(script_name))

println("\n" * "="^60)
println("ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ")
println("="^60)

param_grid = Dict(
    :u0 => [[40.0, 9.0]],                          # фиксируем начальные условия
    :α => [0.05, 0.1, 0.15, 0.2, 0.25],            # варьируем рождаемость жертв
    :β => [0.01, 0.02, 0.03, 0.04],                # варьируем эффективность охоты
    :δ => [0.005, 0.01, 0.015, 0.02],              # варьируем эффективность конверсии
    :γ => [0.2, 0.3, 0.4, 0.5],                    # варьируем смертность хищников
    :tspan => [(0.0, 300.0)],                       # увеличиваем время для анализа
    :solver => [Tsit5()],                            # фиксируем метод решения
    :saveat => [0.2],                                # шаг сохранения
    :experiment_name => ["parametric_scan"]
)

all_params = dict_list(param_grid)

println("Сетка параметров:")
println("  α (рождаемость жертв): ", param_grid[:α])
println("  β (эффективность охоты): ", param_grid[:β])
println("  δ (конверсия): ", param_grid[:δ])
println("  γ (смертность хищников): ", param_grid[:γ])
println("Всего комбинаций: ", length(all_params))

all_results = []  # для хранения сводных результатов
all_dfs = []      # для хранения полных данных (ограниченно)

println("\nЗапуск параметрического сканирования...")

for (i, params) in enumerate(all_params)
    if i % 10 == 0 || i == 1
        println("  Прогресс: $i/$(length(all_params)) | α=$(params[:α]), β=$(params[:β]), δ=$(params[:δ]), γ=$(params[:γ])")
    end

    data, path = produce_or_load(
        datadir(script_name, "parametric_scan"),
        params,
        run_single_experiment,
        prefix = "lv_scan",
        tag = false,
        verbose = false
    )

    result_summary = merge(
        params,
        Dict(
            :x_star => data["x_star"],
            :y_star => data["y_star"],
            :mean_prey => data["mean_prey"],
            :mean_predator => data["mean_predator"],
            :std_prey => data["std_prey"],
            :std_predator => data["std_predator"],
            :dominant_period => data["dominant_period"],
            :theoretical_period => data["theoretical_period"],
            :phase_shift => data["phase_shift"],
            :peak_value_prey => data["peak_value_prey"],
            :peak_value_predator => data["peak_value_predator"],
            :filepath => path
        )
    )
    push!(all_results, result_summary)

    if params[:α] == 0.1 && params[:β] == 0.02 && params[:γ] in [0.2, 0.3, 0.4]
        push!(all_dfs, (params=params, df=data["dataframe"]))
    elseif params[:α] in [0.05, 0.1, 0.2] && params[:β] == 0.02 && params[:γ] == 0.3
        push!(all_dfs, (params=params, df=data["dataframe"]))
    end
end

results_df = DataFrame(all_results)

println("\nПервые 5 строк сводной таблицы результатов:")
println(first(results_df, 5))

CSV.write(datadir(script_name, "parametric_results.csv"), results_df)
println("\nСводные результаты сохранены: ", datadir(script_name, "parametric_results.csv"))

println("\n" * "-"^60)
println("АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ НА СТАЦИОНАРНЫЕ ТОЧКИ")
println("-"^60)

println("\nВлияние γ и δ на равновесную численность жертв x*:")
for γ_val in [0.2, 0.3, 0.4]
    for δ_val in [0.01, 0.02]
        x_star = γ_val / δ_val
        println("  γ=$(γ_val), δ=$(δ_val) -> x* = $(round(x_star, digits=1))")
    end
end

println("\nВлияние α и β на равновесную численность хищников y*:")
for α_val in [0.05, 0.1, 0.2]
    for β_val in [0.01, 0.02, 0.03]
        y_star = α_val / β_val
        println("  α=$(α_val), β=$(β_val) -> y* = $(round(y_star, digits=1))")
    end
end

println("\n" * "-"^60)
println("АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ НА ПЕРИОД КОЛЕБАНИЙ")
println("-"^60)

println("\nСравнение теоретического и фактического периодов:")
subset_df = filter(row -> row.β == 0.02 && row.δ == 0.01, results_df)
sort!(subset_df, [:α, :γ])
for row in eachrow(subset_df)
    println("  α=$(row.α), γ=$(row.γ): T_теор = $(round(row.theoretical_period, digits=2)), T_факт = $(round(row.dominant_period, digits=2))")
end

println("\nПостроение графиков параметрического исследования...")

plot_data = filter(item -> item.params[:β] == 0.02 &&
                           item.params[:γ] == 0.3 &&
                           item.params[:δ] == 0.01, all_dfs)

if !isempty(plot_data)
    plt3 = plot(xlabel="Время", ylabel="Численность жертв",
                title="Влияние α на динамику жертв (β=0.02, γ=0.3, δ=0.01)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))

    for item in plot_data
        α_val = item.params[:α]
        df = item.df
        plot!(plt3, df.t, df.prey, label=L"α = %$α_val")
    end

    savefig(plt3, plotsdir(script_name, "param_alpha_prey.png"))
    println("  Сохранен график: param_alpha_prey.png")

    plt4 = plot(xlabel="Время", ylabel="Численность хищников",
                title="Влияние α на динамику хищников (β=0.02, γ=0.3, δ=0.01)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))

    for item in plot_data
        α_val = item.params[:α]
        df = item.df
        plot!(plt4, df.t, df.predator, label=L"α = %$α_val")
    end

    savefig(plt4, plotsdir(script_name, "param_alpha_predator.png"))
    println("  Сохранен график: param_alpha_predator.png")
end

plot_data = filter(item -> item.params[:α] == 0.1 &&
                           item.params[:β] == 0.02 &&
                           item.params[:δ] == 0.01, all_dfs)

if !isempty(plot_data)
    plt5 = plot(xlabel="Время", ylabel="Численность жертв",
                title="Влияние γ на динамику жертв (α=0.1, β=0.02, δ=0.01)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))

    for item in plot_data
        γ_val = item.params[:γ]
        df = item.df
        plot!(plt5, df.t, df.prey, label=L"γ = %$γ_val")
    end

    savefig(plt5, plotsdir(script_name, "param_gamma_prey.png"))
    println("  Сохранен график: param_gamma_prey.png")

    plt6 = plot(xlabel="Время", ylabel="Численность хищников",
                title="Влияние γ на динамику хищников (α=0.1, β=0.02, δ=0.01)",
                linewidth=2, legend=:topright, grid=true, size=(900, 500))

    for item in plot_data
        γ_val = item.params[:γ]
        df = item.df
        plot!(plt6, df.t, df.predator, label=L"γ = %$γ_val")
    end

    savefig(plt6, plotsdir(script_name, "param_gamma_predator.png"))
    println("  Сохранен график: param_gamma_predator.png")
end

plt7 = scatter(results_df.α, results_df.dominant_period,
               xlabel=L"α (рождаемость жертв)",
               ylabel="Период колебаний",
               title="Зависимость периода от α (γ=0.3 фикс.)",
               markersize=6, markercolor=:blue, alpha=0.7,
               grid=true, size=(800, 500), legend=false)

α_range = 0.05:0.01:0.25
γ_fixed = 0.3
theoretical_T = 2π ./ sqrt.(α_range .* γ_fixed)
plot!(plt7, α_range, theoretical_T,
      linewidth=2, color=:red, linestyle=:dash, label="Теория: 2π/√(αγ)")

savefig(plt7, plotsdir(script_name, "param_period_vs_alpha.png"))
println("  Сохранен график: param_period_vs_alpha.png")

plt8 = scatter(results_df.γ, results_df.dominant_period,
               xlabel=L"γ (смертность хищников)",
               ylabel="Период колебаний",
               title="Зависимость периода от γ (α=0.1 фикс.)",
               markersize=6, markercolor=:red, alpha=0.7,
               grid=true, size=(800, 500), legend=false)

γ_range = 0.2:0.01:0.5
α_fixed = 0.1
theoretical_T_γ = 2π ./ sqrt.(α_fixed .* γ_range)
plot!(plt8, γ_range, theoretical_T_γ,
      linewidth=2, color=:blue, linestyle=:dash, label="Теория: 2π/√(αγ)")

savefig(plt8, plotsdir(script_name, "param_period_vs_gamma.png"))
println("  Сохранен график: param_period_vs_gamma.png")

plt9 = scatter(results_df.α, results_df.std_prey,
               xlabel=L"α (рождаемость жертв)",
               ylabel="Стандартное отклонение численности жертв",
               title="Зависимость амплитуды колебаний от α",
               markersize=6, markercolor=:green, alpha=0.7,
               grid=true, size=(800, 500), legend=false)

savefig(plt9, plotsdir(script_name, "param_amplitude_vs_alpha.png"))
println("  Сохранен график: param_amplitude_vs_alpha.png")

plt10 = scatter(results_df.β, results_df.std_prey,
                xlabel=L"β (эффективность охоты)",
                ylabel="Стандартное отклонение численности жертв",
                title="Зависимость амплитуды колебаний от β",
                markersize=6, markercolor=:orange, alpha=0.7,
                grid=true, size=(800, 500), legend=false)

savefig(plt10, plotsdir(script_name, "param_amplitude_vs_beta.png"))
println("  Сохранен график: param_amplitude_vs_beta.png")

plt11 = plot(xlabel="Популяция жертв (x)", ylabel="Популяция хищников (y)",
             title="Сравнение фазовых портретов (разные γ)",
             legend=:topright, grid=true, size=(800, 600))

colors = [:blue, :red, :green, :orange]
for (idx, item) in enumerate(plot_data)
    γ_val = item.params[:γ]
    df = item.df
    plot!(plt11, df.prey, df.predator,
          label=L"γ = %$γ_val", color=colors[idx],
          linewidth=1.5)
end

savefig(plt11, plotsdir(script_name, "param_phase_comparison.png"))
println("  Сохранен график: param_phase_comparison.png")

using StatsPlots

α_unique = sort(unique(results_df.α))
γ_unique = sort(unique(results_df.γ))

period_matrix = zeros(length(α_unique), length(γ_unique))
for i, α_val) in enumerate(α_unique)
    for (j, γ_val) in enumerate(γ_unique)
        rows = filter(row -> row.α == α_val && row.γ == γ_val, results_df)
        if nrow(rows) > 0
            period_matrix[i, j] = mean(rows.dominant_period)
        end
    end
end

plt12 = heatmap(γ_unique, α_unique, period_matrix,
                xlabel="γ (смертность хищников)",
                ylabel="α (рождаемость жертв)",
                title="Тепловая карта: период колебаний",
                color=:viridis, colorbar_title="Период",
                size=(800, 600))

savefig(plt12, plotsdir(script_name, "param_period_heatmap.png"))
println("  Сохранен график: param_period_heatmap.png")

println("\n" * "-"^60)
println("АНАЛИЗ УСТОЙЧИВОСТИ СИСТЕМЫ")
println("-"^60)

results_df[!, :cv_prey] = results_df.std_prey ./ results_df.mean_prey
results_df[!, :cv_predator] = results_df.std_predator ./ results_df.mean_predator

println("\nКоэффициент вариации (CV = σ/μ) для разных параметров:")
println("  CV жертв: от $(round(minimum(results_df.cv_prey), digits=2)) до $(round(maximum(results_df.cv_prey), digits=2))")
println("  CV хищников: от $(round(minimum(results_df.cv_predator), digits=2)) до $(round(maximum(results_df.cv_predator), digits=2))")

most_stable = sort(results_df, :cv_prey)[1:3, [:α, :β, :γ, :δ, :cv_prey]]
println("\nНаиболее устойчивые режимы (минимальные колебания жертв):")
for row in eachrow(most_stable)
    println("  α=$(row.α), β=$(row.β), γ=$(row.γ), δ=$(row.δ): CV = $(round(row.cv_prey, digits=3))")
end

least_stable = sort(results_df, :cv_prey, rev=true)[1:3, [:α, :β, :γ, :δ, :cv_prey]]
println("\nНаименее устойчивые режимы (максимальные колебания жертв):")
for row in eachrow(least_stable)
    println("  α=$(row.α), β=$(row.β), γ=$(row.γ), δ=$(row.δ): CV = $(round(row.cv_prey, digits=3))")
end

println("\n" * "="^60)
println("БЕНЧМАРКИНГ ПРОИЗВОДИТЕЛЬНОСТИ")
println("="^60)

benchmark_results = []

for α_val in [0.05, 0.1, 0.2]
    bench_params = Dict(
        :u0 => [40.0, 9.0],
        :α => α_val,
        :β => 0.02,
        :δ => 0.01,
        :γ => 0.3,
        :tspan => (0.0, 200.0),
        :solver => Tsit5(),
        :saveat => 0.1
    )

    function bench_run()
        prob = ODEProblem(lotka_volterra!, bench_params[:u0], bench_params[:tspan],
                         [bench_params[:α], bench_params[:β], bench_params[:δ], bench_params[:γ]])
        return solve(prob, bench_params[:solver]; saveat=bench_params[:saveat],
                    reltol=1e-8, abstol=1e-10)
    end

    println("\nБенчмарк для α = $α_val:")
    b = @benchmark $bench_run() samples=30 evals=1
    time_ms = median(b).time / 1e6  # время в миллисекундах
    push!(benchmark_results, (α=α_val, time_ms=time_ms))
    println("  Среднее время: ", round(time_ms, digits=2), " мс")
end

bench_df = DataFrame(benchmark_results)

@save datadir(script_name, "all_results.jld2")
    base_params param_grid all_params results_df bench_df df_base

@save datadir(script_name, "all_plots.jld2")
    plt1 plt2 plt3 plt4 plt5 plt6 plt7 plt8 plt9 plt10 plt11 plt12
