using StableRNGs
using Distributions
using ConcurrentSim
using ResumableFunctions
using Random
using DataFrames
using Plots
using StatsBase
using PrettyTables
using CSV
using StatsPlots

# Установка seed для воспроизводимости
const GLOBAL_RNG = StableRNG(42)

# Создание директории для графиков
if !isdir("plots")
    mkdir("plots")
end
if !isdir("results")
    mkdir("results")
end

# ============================================================================
# Часть 1: Модель М/М/с
# ============================================================================

struct MMC_Model
    λ::Float64          # интенсивность входного потока
    μ::Float64          # интенсивность обслуживания одного канала
    c::Int              # число каналов
    num_customers::Int  # число заявок для моделирования
    seed::Int           # seed для RNG
end

struct MMC_Results
    arrival_times::Vector{Float64}
    service_start_times::Vector{Float64}
    service_end_times::Vector{Float64}
    waiting_times::Vector{Float64}
    service_times::Vector{Float64}
    system_times::Vector{Float64}
    server_utilization::Float64
end

# Определение поведения клиента
@resumable function mmc_customer(
    env::Environment, 
    server::Resource, 
    id::Integer, 
    arrival_time::Float64,
    service_dist::Distribution,
    rng::AbstractRNG,
    results_channel::Channel
)
    # Ожидание прибытия
    @yield timeout(env, arrival_time)
    arrival = now(env)
    
    # Запрос сервера
    @yield request(server)
    service_start = now(env)
    
    # Обслуживание
    service_time = rand(rng, service_dist)
    @yield timeout(env, service_time)
    
    # Освобождение сервера
    @yield release(server)
    service_end = now(env)
    
    # Отправка данных
    put!(results_channel, (arrival, service_start, service_end, service_time))
end

function run_mmc_simulation(model::MMC_Model)
    rng = StableRNG(model.seed)
    
    sim = Simulation()
    server = Resource(sim, model.c)
    
    arrival_dist = Exponential(1 / model.λ)
    service_dist = Exponential(1 / model.μ)
    
    arrivals = Float64[]
    service_starts = Float64[]
    service_ends = Float64[]
    service_times = Float64[]
    
    # Канал для сбора результатов
    results_channel = Channel{Tuple{Float64,Float64,Float64,Float64}}(model.num_customers)
    
    # Создание процессов клиентов
    arrival_time = 0.0
    
    for i in 1:model.num_customers
        arrival_time += rand(rng, arrival_dist)
        @process mmc_customer(sim, server, i, arrival_time, service_dist, rng, results_channel)
    end
    
    # Запуск симуляции
    run(sim)
    
    # Сбор результатов
    close(results_channel)
    for result in results_channel
        push!(arrivals, result[1])
        push!(service_starts, result[2])
        push!(service_ends, result[3])
        push!(service_times, result[4])
    end
    
    if isempty(arrivals)
        return MMC_Results([], [], [], [], [], [], 0.0)
    end
    
    waiting_times = service_starts .- arrivals
    system_times = service_ends .- arrivals
    
    # Расчет загрузки серверов
    total_busy_time = sum(service_times)
    total_time = isempty(service_ends) ? 1.0 : maximum(service_ends)
    server_utilization = total_busy_time / (total_time * model.c)
    
    return MMC_Results(
        arrivals, service_starts, service_ends,
        waiting_times, service_times, system_times,
        server_utilization
    )
end

function analytical_mmc(λ::Float64, μ::Float64, c::Int)
    ρ = λ / (c * μ)
    
    if ρ >= 1
        error("Система нестационарна: ρ = $ρ >= 1")
    end
    
    # Расчет P0
    sum1 = sum((c * ρ)^n / factorial(n) for n in 0:c-1)
    sum2 = (c * ρ)^c / (factorial(c) * (1 - ρ))
    P0 = 1 / (sum1 + sum2)
    
    # Вероятность ожидания (формула Эрланга)
    Pwait = ((c * ρ)^c / (factorial(c) * (1 - ρ))) * P0
    
    # Среднее число в очереди
    Lq = (ρ / (1 - ρ)) * Pwait
    
    # Время ожидания
    Wq = Lq / λ
    W = Wq + 1/μ
    L = λ * W
    
    return (ρ=ρ, P0=P0, Pwait=Pwait, Lq=Lq, Wq=Wq, W=W, L=L)
end

function plot_mmc_results(results::MMC_Results, analytical, save_path::String="plots/mmc_results.png")
    if isempty(results.waiting_times)
        println("Нет данных для построения графиков MMC")
        return nothing
    end
    
    # График 1: Времена ожидания
    p1 = histogram(results.waiting_times, 
        bins=30, 
        title="Распределение времени ожидания",
        xlabel="Время ожидания", 
        ylabel="Частота",
        legend=false,
        color=:blue,
        alpha=0.7
    )
    
    # График 2: Времена в системе
    p2 = histogram(results.system_times, 
        bins=30, 
        title="Распределение времени в системе",
        xlabel="Время в системе", 
        ylabel="Частота",
        legend=false,
        color=:green,
        alpha=0.7
    )
    
    # График 3: Сравнение с аналитикой
    p3 = bar(
        ["Wq", "W"],
        [mean(results.waiting_times), mean(results.system_times)],
        title="Сравнение симуляции и аналитики",
        ylabel="Время",
        label="Симуляция",
        color=:orange
    )
    bar!(
        ["Wq", "W"],
        [analytical.Wq, analytical.W],
        label="Аналитика",
        color=:red,
        alpha=0.6
    )
    
    # Объединение графиков
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000))
    
    # Сохранение графика
    savefig(final_plot, save_path)
    println("График сохранен: $save_path")
    
    # Сохранение данных в CSV
    df = DataFrame(
        waiting_time=results.waiting_times,
        system_time=results.system_times,
        service_time=results.service_times
    )
    CSV.write("results/mmc_data.csv", df)
    println("Данные сохранены: results/mmc_data.csv")
    
    return final_plot
end

# ============================================================================
# Часть 2: Модель Росса
# ============================================================================

struct RossModel
    N::Int              # количество основных машин
    S::Int              # количество запасных машин
    λ::Float64          # среднее время безотказной работы
    μ::Float64          # среднее время ремонта
    num_repairmen::Int  # количество ремонтников
    runs::Int           # количество прогонов
    seed::Int
end

struct RossResults
    crash_times::Vector{Float64}
    average_crash_time::Float64
    std_crash_time::Float64
    median_crash_time::Float64
    min_crash_time::Float64
    max_crash_time::Float64
end

function run_ross_single(model::RossModel, sim_id::Int)
    rng = StableRNG(model.seed + sim_id)
    
    working_machines = model.N  # работающие машины
    spare_machines = model.S     # резервные машины
    broken_machines = 0           # сломанные машины (в ремонте)
    repair_queue = 0              # очередь на ремонт
    active_repairs = 0            # активные ремонты
    
    time = 0.0
    failure_dist = Exponential(1/model.λ)  # Исправлено: λ - это среднее время
    repair_dist = Exponential(1/model.μ)    # Исправлено: μ - это среднее время
    
    # События
    if working_machines > 0
        next_failure_time = rand(rng, failure_dist)
    else
        next_failure_time = Inf
    end
    next_repair_time = Inf
    repair_completion_times = Float64[]
    
    while true
        # Проверяем, какое событие произойдет раньше
        if next_failure_time < next_repair_time
            # Отказ машины
            time = next_failure_time
            
            if spare_machines > 0
                # Есть резерв - берем его
                spare_machines -= 1
                broken_machines += 1
                
                # Отправляем в ремонт
                if active_repairs < model.num_repairmen
                    # Ремонтник свободен - начинаем ремонт
                    active_repairs += 1
                    repair_time = rand(rng, repair_dist)
                    push!(repair_completion_times, time + repair_time)
                    next_repair_time = minimum(repair_completion_times)
                else
                    # Все ремонтники заняты - ставим в очередь
                    repair_queue += 1
                end
                
                # Планируем следующий отказ
                if working_machines > 0
                    next_failure_time = time + rand(rng, failure_dist)
                else
                    next_failure_time = Inf
                end
            else
                # Нет резерва - система падает
                return time
            end
        elseif isfinite(next_repair_time)
            # Завершение ремонта
            time = next_repair_time
            
            # Удаляем завершенные ремонты
            repair_completion_times = [t for t in repair_completion_times if t > time + 1e-10]
            
            broken_machines -= 1
            active_repairs -= 1
            
            if broken_machines < 0
                broken_machines = 0
            end
            
            # Возвращаем машину в работу или резерв
            if spare_machines + working_machines < model.N
                working_machines += 1
            else
                spare_machines += 1
            end
            
            # Начинаем следующий ремонт из очереди
            if repair_queue > 0
                repair_queue -= 1
                active_repairs += 1
                repair_time = rand(rng, repair_dist)
                push!(repair_completion_times, time + repair_time)
            end
            
            # Планируем следующий ремонт
            if !isempty(repair_completion_times)
                next_repair_time = minimum(repair_completion_times)
            else
                next_repair_time = Inf
            end
            
            # Планируем следующий отказ
            if working_machines > 0
                next_failure_time = time + rand(rng, failure_dist)
            else
                next_failure_time = Inf
            end
        else
            # Нет событий - система работает бесконечно
            return Inf
        end
    end
end

function run_ross_multiple(model::RossModel)
    println("\n=== Запуск модели Росса ===")
    println("Параметры: N=$(model.N), S=$(model.S), λ=$(model.λ), μ=$(model.μ), ремонтников=$(model.num_repairmen)")
    println("Прогонов: $(model.runs)")
    
    crash_times = Float64[]
    
    for i in 1:model.runs
        crash_time = run_ross_single(model, i)
        push!(crash_times, crash_time)
        println("  Прогон $i: время до падения = $(round(crash_time, digits=2))")
    end
    
    # Фильтрация бесконечных значений
    finite_times = [t for t in crash_times if isfinite(t)]
    
    if isempty(finite_times)
        avg_crash = Inf
        std_crash = 0.0
        median_crash = Inf
        min_crash = Inf
        max_crash = -Inf
    else
        avg_crash = mean(finite_times)
        std_crash = std(finite_times)
        median_crash = median(finite_times)
        min_crash = minimum(finite_times)
        max_crash = maximum(finite_times)
    end
    
    println("\nРезультаты:")
    println("  Среднее время до падения: $(round(avg_crash, digits=2)) ± $(round(std_crash, digits=2))")
    println("  Медиана: $(round(median_crash, digits=2))")
    println("  Минимум: $(round(min_crash, digits=2))")
    println("  Максимум: $(round(max_crash, digits=2))")
    
    return RossResults(crash_times, avg_crash, std_crash, median_crash, min_crash, max_crash)
end

function plot_ross_results(results::RossResults, model::RossModel, save_path::String="plots/ross_results.png")
    # График 1: Гистограмма времен до падения
    p1 = histogram(results.crash_times, 
        bins=15,
        title="Распределение времени до падения системы\nN=$(model.N), S=$(model.S), ремонтников=$(model.num_repairmen)",
        xlabel="Время до падения",
        ylabel="Частота",
        legend=false,
        color=:red,
        alpha=0.7
    )
    vline!([results.average_crash_time], label="Среднее", color=:blue, linestyle=:dash, linewidth=2)
    
    # График 2: Времена до падения по прогонам
    p2 = plot(results.crash_times,
        marker=:circle,
        line=:stem,
        title="Время до падения по прогонам",
        xlabel="Номер прогона",
        ylabel="Время до падения",
        legend=false,
        color=:blue
    )
    hline!([results.average_crash_time], label="Среднее", color=:red, linestyle=:dash, linewidth=2)
    
    # График 3: Ящик с усами (boxplot) с использованием StatsPlots
    p3 = boxplot(["Система"], results.crash_times,
        title="Ящик с усами (Boxplot) времени до падения",
        ylabel="Время до падения",
        color=:orange,
        fillalpha=0.7
    )
    
    # Объединение графиков
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000))
    
    # Сохранение графика
    savefig(final_plot, save_path)
    println("График сохранен: $save_path")
    
    # Сохранение данных в CSV
    df = DataFrame(crash_time=results.crash_times)
    CSV.write("results/ross_data_N$(model.N)_S$(model.S)_R$(model.num_repairmen).csv", df)
    println("Данные сохранены: results/ross_data_N$(model.N)_S$(model.S)_R$(model.num_repairmen).csv")
    
    return final_plot
end

# ============================================================================
# Часть 3: Основная программа и анализ параметров
# ============================================================================

function run_parameter_study()
    println("\n" * "="^60)
    println("ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПАРАМЕТРОВ НА МОДЕЛЬ РОССА")
    println("="^60)
    
    param_study = [
        (N=10, S=3, name="Базовая конфигурация"),
        (N=10, S=5, name="Увеличенный резерв"),
        (N=10, S=10, name="Большой резерв"),
        (N=20, S=5, name="Больше машин"),
        (N=5, S=3, name="Меньше машин"),
    ]
    
    results_table = []
    
    for params in param_study
        println("\n--- $(params.name) ---")
        println("N=$(params.N), S=$(params.S)")
        
        model = RossModel(
            params.N, params.S, 100.0, 1.0, 1, 20, 42
        )
        
        results = run_ross_multiple(model)
        
        # Сохранение графика для этой конфигурации
        plot_path = "plots/ross_study_N$(params.N)_S$(params.S).png"
        plot_ross_results(results, model, plot_path)
        
        push!(results_table, (
            Name=params.name,
            N=params.N,
            S=params.S,
            AvgTime=round(results.average_crash_time, digits=2),
            StdDev=round(results.std_crash_time, digits=2),
            Median=round(results.median_crash_time, digits=2),
            Min=round(results.min_crash_time, digits=2),
            Max=round(results.max_crash_time, digits=2)
        ))
    end
    
    println("\n" * "="^60)
    println("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    println("="^60)
    pretty_table(results_table)
    
    # Сохранение таблицы в файл
    df_table = DataFrame(results_table)
    CSV.write("results/parameter_study.csv", df_table)
    println("Таблица сохранена: results/parameter_study.csv")
    
    # График сравнения конфигураций
    p_comparison = bar(
        [r.Name for r in results_table],
        [r.AvgTime for r in results_table],
        title="Сравнение среднего времени до падения",
        ylabel="Среднее время до падения",
        xlabel="Конфигурация",
        legend=false,
        color=:purple
    )
    savefig(p_comparison, "plots/ross_configurations_comparison.png")
    println("График сравнения сохранен: plots/ross_configurations_comparison.png")
    
    return results_table
end

function run_with_multiple_repairmen()
    println("\n" * "="^60)
    println("ИССЛЕДОВАНИЕ ВЛИЯНИЯ КОЛИЧЕСТВА РЕМОНТНИКОВ")
    println("="^60)
    
    repairmen_counts = [1, 2, 3, 4, 5]
    results = []
    
    for r in repairmen_counts
        println("\nРемонтников: $r")
        
        model = RossModel(10, 3, 100.0, 1.0, r, 20, 42)
        sim_results = run_ross_multiple(model)
        
        # Сохранение графика
        plot_path = "plots/ross_repairmen_R$r.png"
        plot_ross_results(sim_results, model, plot_path)
        
        push!(results, (
            Repairmen=r,
            AvgCrashTime=round(sim_results.average_crash_time, digits=2),
            StdDev=round(sim_results.std_crash_time, digits=2),
            Median=round(sim_results.median_crash_time, digits=2),
            Min=round(sim_results.min_crash_time, digits=2),
            Max=round(sim_results.max_crash_time, digits=2)
        ))
    end
    
    println("\n" * "="^60)
    println("СВОДНАЯ ТАБЛИЦА")
    println("="^60)
    pretty_table(results)
    
    # Сохранение таблицы в файл
    df_results = DataFrame(results)
    CSV.write("results/repairmen_study.csv", df_results)
    println("Таблица сохранена: results/repairmen_study.csv")
    
    # График зависимости
    p_dependency = plot(
        [r.Repairmen for r in results], 
        [r.AvgCrashTime for r in results],
        marker=:circle,
        markersize=8,
        linewidth=2,
        title="Зависимость времени до падения от числа ремонтников",
        xlabel="Количество ремонтников",
        ylabel="Среднее время до падения",
        legend=false,
        color=:purple
    )
    savefig(p_dependency, "plots/ross_dependency_repairmen.png")
    println("График зависимости сохранен: plots/ross_dependency_repairmen.png")
    display(p_dependency)
    
    return results
end

function generate_report()
    println("\n" * "="^60)
    println("ГЕНЕРАЦИЯ ОТЧЕТА")
    println("="^60)
    
    report_path = "results/simulation_report.txt"
    
    open(report_path, "w") do io
        println(io, "="^70)
        println(io, "ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №7")
        println(io, "ДИСКРЕТНО-СОБЫТИЙНОЕ МОДЕЛИРОВАНИЕ")
        println(io, "="^70)
        println(io, "\nДата и время: ", Dates.now())
        println(io, "\n" * "-"^50)
        println(io, "РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
        println(io, "-"^50)
        
        println(io, "\nВсе графики сохранены в директории 'plots/'")
        println(io, "Все данные сохранены в директории 'results/'")
        println(io, "\nСписок сохраненных файлов:")
        
        if isdir("plots")
            println(io, "\nГрафики:")
            for file in readdir("plots")
                println(io, "  - plots/$file")
            end
        end
        
        if isdir("results")
            println(io, "\nДанные:")
            for file in readdir("results")
                println(io, "  - results/$file")
            end
        end
    end
    
    println("Отчет сохранен: $report_path")
end

# ============================================================================
# ЗАПУСК ВСЕХ ЗАДАНИЙ
# ============================================================================

function main()
    println("="^70)
    println("ЛАБОРАТОРНАЯ РАБОТА №7: ИМИТАЦИОННОЕ МОДЕЛИРОВАНИЕ")
    println("="^70)
    
    # ------------------------------------------------------------------------
    # 1. Модель М/М/с
    # ------------------------------------------------------------------------
    println("\n" * "-"^50)
    println("1. МОДЕЛЬ М/М/с")
    println("-"^50)
    
    try
        mmc_model = MMC_Model(0.9, 0.5, 2, 200, 42)
        mmc_results = run_mmc_simulation(mmc_model)
        mmc_analytical = analytical_mmc(mmc_model.λ, mmc_model.μ, mmc_model.c)
        
        mmc_plot = plot_mmc_results(mmc_results, mmc_analytical, "plots/mmc_results.png")
        if mmc_plot !== nothing
            display(mmc_plot)
        end
        
        # Сохранение статистики
        mmc_stats = DataFrame(
            Parameter=["λ", "μ", "c", "ρ", "P0", "Pwait", "Wq_sim", "Wq_analytical", "W_sim", "W_analytical", "server_utilization"],
            Value=[mmc_model.λ, mmc_model.μ, mmc_model.c, mmc_analytical.ρ, mmc_analytical.P0, mmc_analytical.Pwait,
                   mean(mmc_results.waiting_times), mmc_analytical.Wq,
                   mean(mmc_results.system_times), mmc_analytical.W,
                   mmc_results.server_utilization]
        )
        CSV.write("results/mmc_statistics.csv", mmc_stats)
        println("Статистика MMC сохранена: results/mmc_statistics.csv")
        
    catch e
        println("Ошибка в модели М/М/с: $e")
        println(stacktrace(catch_backtrace()))
    end
    
    # ------------------------------------------------------------------------
    # 2. Модель Росса (базовая)
    # ------------------------------------------------------------------------
    println("\n" * "-"^50)
    println("2. МОДЕЛЬ РОССА (БАЗОВАЯ)")
    println("-"^50)
    
    ross_model_basic = RossModel(10, 3, 100.0, 1.0, 1, 30, 42)
    ross_results_basic = run_ross_multiple(ross_model_basic)
    ross_plot_basic = plot_ross_results(ross_results_basic, ross_model_basic, "plots/ross_basic.png")
    display(ross_plot_basic)
    
    # ------------------------------------------------------------------------
    # 3. Модель Росса с несколькими ремонтниками
    # ------------------------------------------------------------------------
    println("\n" * "-"^50)
    println("3. МОДЕЛЬ РОССА (НЕСКОЛЬКО РЕМОНТНИКОВ)")
    println("-"^50)
    
    for r in [2, 3, 4]
        println("\n--- $r ремонтника(ов) ---")
        model = RossModel(10, 3, 100.0, 1.0, r, 20, 42)
        results = run_ross_multiple(model)
        plot_ross_results(results, model, "plots/ross_repairmen_R$r.png")
    end
    
    # ------------------------------------------------------------------------
    # 4. Исследование параметров
    # ------------------------------------------------------------------------
    println("\n" * "-"^50)
    println("4. ИССЛЕДОВАНИЕ ПАРАМЕТРОВ")
    println("-"^50)
    
    param_table = run_parameter_study()
    repairmen_study = run_with_multiple_repairmen()
    
    # ------------------------------------------------------------------------
    # 5. Генерация отчета
    # ------------------------------------------------------------------------
    generate_report()
    
    println("\n" * "="^70)
    println("ВЫПОЛНЕНИЕ ЗАДАНИЙ ЗАВЕРШЕНО")
    println("="^70)
    println("\nРезультаты сохранены в:")
    println("  - Графики: ./plots/")
    println("  - Данные: ./results/")
    println("  - Отчет: ./results/simulation_report.txt")
end

# Запуск основной программы
main()
