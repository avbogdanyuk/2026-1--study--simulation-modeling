using DrWatson
@quickactivate "project"
using BlackBoxOptim, Random, Statistics

include(srcdir("sir_model.jl"))

function cost_multi(x)

    model = initialize_sir(;
        Ns = [1000, 1000, 1000],                    # три города по 1000 жителей
        β_und = fill(x[1], 3),                      # заразность невыявленных (одинакова для всех городов)
        β_det = fill(x[1]/10, 3),                   # заразность выявленных (в 10 раз ниже)
        infection_period = 14,                      # длительность болезни (дней)
        detection_time = round(Int, x[2]),          # время до выявления (целое число дней)
        death_rate = x[3],                          # вероятность смерти при завершении болезни
        reinfection_probability = 0.1,              # вероятность повторного заражения
        Is = [0, 0, 1],                             # начальные заражённые (только в третьем городе)
        seed = 42,                                  # зерно случайных чисел
        n_steps = 100,                              # длительность симуляции (дней)
    )

    infected_frac(model) = count(a.status == :I for a in allagents(model)) / nagents(model)
    dead_count(model) = 3000 - nagents(model)       # 3000 = сумма Ns

    replicates = 5
    peak_vals = Float64[]    # массив для хранения пиковых значений заболеваемости
    dead_vals = Int[]        # массив для хранения количества умерших

    for rep = 1:replicates

        model = initialize_sir(;
            Ns = [1000, 1000, 1000],
            β_und = fill(x[1], 3),
            β_det = fill(x[1]/10, 3),
            infection_period = 14,
            detection_time = round(Int, x[2]),
            death_rate = x[3],
            reinfection_probability = 0.1,
            Is = [0, 0, 1],
            seed = 42 + rep,                       # увеличиваем зерно для разнообразия
            n_steps = 100,
        )

        for step = 1:100
            Agents.step!(model, 1)
            frac = infected_frac(model)
            if frac > peak_infected
                peak_infected = frac
            end
        end

        push!(peak_vals, peak_infected)
        push!(dead_vals, dead_count(model))
    end

    return (mean(peak_vals), mean(dead_vals) / 3000)
end

result = bboptimize(
    cost_multi,
    Method = :borg_moea,                                    # метод оптимизации
    FitnessScheme = ParetoFitnessScheme{2}(is_minimizing = true),  # два критерия на минимизацию
    SearchRange = [
        (0.1, 1.0),      # β_und — коэффициент заражения
        (3.0, 14.0),     # detection_time — время до выявления (дни)
        (0.01, 0.1),     # death_rate — вероятность смерти
    ],
    NumDimensions = 3,                                       # количество оптимизируемых параметров
    MaxTime = 120,                                           # максимальное время выполнения (120 секунд = 2 минуты)
    TraceMode = :compact,                                    # компактный вывод прогресса
)

best = best_candidate(result)      # оптимальный вектор параметров
fitness = best_fitness(result)     # соответствующие значения критериев

println("Оптимальные параметры:")
println("β_und = $(best[1])")
println("Время выявления = $(round(Int, best[2])) дней")
println("Смертность = $(best[3])")

println("Достигнутые показатели:")
println("Пик заболеваемости: $(fitness[1])")
println("Доля умерших: $(fitness[2])")

save(datadir("optimization_result.jld2"), Dict("best" => best, "fitness" => fitness))
