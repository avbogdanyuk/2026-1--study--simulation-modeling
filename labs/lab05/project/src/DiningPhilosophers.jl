# Модуль DiningPhilosophers - моделирование задачи "Обедающие философы" с помощью сетей Петри
"""
Данный модуль реализует аппарат сетей Петри для моделирования классической задачи
синхронизации "Обедающие философы". Включает построение сетей, детерминированное и
стохастическое моделирование, обнаружение deadlock и визуализацию.
"""

module DiningPhilosophers

using OrdinaryDiffEq
using Plots
using DataFrames
using Random, LinearAlgebra

export build_classical_network, build_arbiter_network
export simulate_ode, simulate_stochastic
export detect_deadlock, plot_marking_evolution

# Определение структуры сети Петри

"""
Структура, представляющая сеть Петри.

Поля:
- n_places::Int - количество позиций (мест)
- n_transitions::Int - количество переходов
- incidence::Matrix{Int} - матрица инцидентности (places × transitions)
- place_names::Vector{Symbol} - имена позиций
- transition_names::Vector{Symbol} - имена переходов
"""

struct PetriNet
    n_places::Int
    n_transitions::Int
    incidence::Matrix{Int}
    place_names::Vector{Symbol}
    transition_names::Vector{Symbol}
end

"""
PetriNet(n_places, n_transitions; place_names, transition_names)

Конструктор сети Петри. Создаёт пустую сеть с заданным количеством позиций и переходов.
Если имена не указаны, генерируются автоматически (p1, p2, ... и t1, t2, ...).
"""

function PetriNet(
    n_places,
    n_transitions;
    place_names = Symbol[],
    transition_names = Symbol[],
)
    incidence = zeros(Int, n_places, n_transitions)
    if isempty(place_names)
        place_names = [Symbol("p$i") for i = 1:n_places]
    end
    if isempty(transition_names)
        transition_names = [Symbol("t$i") for i = 1:n_transitions]
    end
    PetriNet(n_places, n_transitions, incidence, place_names, transition_names)
end

"""
add_arc!(net, place, transition, sign)

Добавляет дугу в сеть Петри.
- sign = -1: входная дуга (от позиции к переходу)
- sign = +1: выходная дуга (от перехода к позиции)
"""

function add_arc!(net::PetriNet, place::Int, transition::Int, sign::Int)
    net.incidence[place, transition] += sign
end

# Построение сетей Петри для задачи "Обедающие философы"

"""
build_classical_network(N)

Строит классическую сеть Петри для N философов.

Структура сети:
- Позиции: Think_i, Hungry_i, Eat_i, Fork_i для каждого философа i
- Переходы: GetLeft_i, GetRight_i, PutForks_i для каждого философа i

Правила срабатывания:
1. GetLeft_i: Think_i + Fork_left -> Hungry_i
2. GetRight_i: Hungry_i + Fork_right -> Eat_i
3. PutForks_i: Eat_i -> Think_i + Fork_left + Fork_right

Возвращает:
- net: сеть Петри
- u0: начальная маркировка (все философы думают, все вилки свободны)
- place_names: имена позиций
"""

function build_classical_network(N::Int)
    n_places = 4N
    n_transitions = 3N
    net = PetriNet(n_places, n_transitions)
    
    # Именование позиций
    for i = 1:N
        net.place_names[i] = Symbol("Think_$i")
        net.place_names[N+i] = Symbol("Hungry_$i")
        net.place_names[2N+i] = Symbol("Eat_$i")
        net.place_names[3N+i] = Symbol("Fork_$i")
    end
    
    # Именование переходов
    for i = 1:N
        net.transition_names[i] = Symbol("GetLeft_$i")
        net.transition_names[N+i] = Symbol("GetRight_$i")
        net.transition_names[2N+i] = Symbol("PutForks_$i")
    end
    
    # Построение дуг
    for i = 1:N
        think = i
        hungry = N + i
        eat = 2N + i
        left_fork = 3N + i
        right_fork = 3N + (i % N + 1)
        get_left = i
        get_right = N + i
        put_forks = 2N + i
        
        # GetLeft: из Think и левой вилки в Hungry
        add_arc!(net, think, get_left, -1)
        add_arc!(net, left_fork, get_left, -1)
        add_arc!(net, hungry, get_left, +1)
        
        # GetRight: из Hungry и правой вилки в Eat
        add_arc!(net, hungry, get_right, -1)
        add_arc!(net, right_fork, get_right, -1)
        add_arc!(net, eat, get_right, +1)
        
        # PutForks: из Eat в Think и обе вилки
        add_arc!(net, eat, put_forks, -1)
        add_arc!(net, think, put_forks, +1)
        add_arc!(net, left_fork, put_forks, +1)
        add_arc!(net, right_fork, put_forks, +1)
    end
    
    # Начальная маркировка
    u0 = zeros(Float64, n_places)
    for i = 1:N
        u0[i] = 1.0       # все философы думают
        u0[3N+i] = 1.0    # все вилки свободны
    end
    
    return net, u0, net.place_names
end

"""
build_arbiter_network(N)

Строит сеть Петри с арбитром для N философов.

Отличие от классической сети: добавляется позиция Arbiter, которая ограничивает
количество одновременно обедающих философов до N-1. Арбитр инициализируется с N-1 фишками.

Возвращает:
- net: сеть Петри
- u0: начальная маркировка
- place_names: имена позиций
"""

function build_arbiter_network(N::Int)
    n_places = 4N + 1
    n_transitions = 3N
    net = PetriNet(n_places, n_transitions)
    
    # Именование позиций
    for i = 1:N
        net.place_names[i] = Symbol("Think_$i")
        net.place_names[N+i] = Symbol("Hungry_$i")
        net.place_names[2N+i] = Symbol("Eat_$i")
        net.place_names[3N+i] = Symbol("Fork_$i")
    end
    net.place_names[4N+1] = :Arbiter
    
    # Именование переходов
    for i = 1:N
        net.transition_names[i] = Symbol("GetLeft_$i")
        net.transition_names[N+i] = Symbol("GetRight_$i")
        net.transition_names[2N+i] = Symbol("PutForks_$i")
    end
    
    arbiter_idx = 4N + 1
    
    # Построение дуг
    for i = 1:N
        think = i
        hungry = N + i
        eat = 2N + i
        left_fork = 3N + i
        right_fork = 3N + (i % N + 1)
        get_left = i
        get_right = N + i
        put_forks = 2N + i
        
        # GetLeft: Think + левая вилка + арбитр -> Hungry
        add_arc!(net, think, get_left, -1)
        add_arc!(net, left_fork, get_left, -1)
        add_arc!(net, arbiter_idx, get_left, -1)
        add_arc!(net, hungry, get_left, +1)
        
        # GetRight: Hungry + правая вилка -> Eat
        add_arc!(net, hungry, get_right, -1)
        add_arc!(net, right_fork, get_right, -1)
        add_arc!(net, eat, get_right, +1)
        
        # PutForks: Eat -> Think + обе вилки + арбитр
        add_arc!(net, eat, put_forks, -1)
        add_arc!(net, think, put_forks, +1)
        add_arc!(net, left_fork, put_forks, +1)
        add_arc!(net, right_fork, put_forks, +1)
        add_arc!(net, arbiter_idx, put_forks, +1)
    end
    
    # Начальная маркировка
    u0 = zeros(Float64, n_places)
    for i = 1:N
        u0[i] = 1.0
        u0[3N+i] = 1.0
    end
    u0[arbiter_idx] = N - 1   # арбитр с N-1 фишками
    
    return net, u0, net.place_names
end

# Детерминированное моделирование (ODE)

"""
vectorfield(net, rates)

Создаёт функцию векторного поля для системы ОДУ, описывающей сеть Петри.

Скорость срабатывания перехода j: rates[j] * ∏_{i: входная дуга} u[i]^{кратность}
"""

function vectorfield(net::PetriNet, rates = ones(net.n_transitions))
    function f!(du, u, params, t)
        a = zeros(net.n_transitions)
        for j = 1:net.n_transitions
            rate = rates[j]
            prod = rate
            for i = 1:net.n_places
                if net.incidence[i, j] < 0
                    prod *= u[i] ^ (-net.incidence[i, j])
                end
            end
            a[j] = prod
        end
        du .= net.incidence * a
    end
    return f!
end

"""
simulate_ode(net, u0, tmax; saveat)

Выполняет детерминированное моделирование сети Петри с помощью решения ОДУ.

Аргументы:
- net: сеть Петри
- u0: начальная маркировка
- tmax: максимальное время моделирования
- saveat: шаг сохранения результатов

Возвращает: DataFrame с временем и маркировками
"""

function simulate_ode(net::PetriNet, u0::Vector{Float64}, tmax::Float64; saveat = 0.1)
    f = vectorfield(net)
    prob = ODEProblem(f, u0, (0.0, tmax))
    sol = solve(prob, Tsit5(), saveat = saveat)
    
    df = DataFrame(time = sol.t)
    for i = 1:net.n_places
        df[!, String(net.place_names[i])] = sol[i, :]
    end
    return df
end

# Стохастическое моделирование (алгоритм Гиллеспи)

"""
    simulate_stochastic(net, u0, tmax; rates, rng)

Выполняет стохастическое моделирование сети Петри с использованием алгоритма Гиллеспи.

Алгоритм:
1. Вычисляются скорости всех переходов a[j]
2. Время до следующего события: dt = -log(rand()) / sum(a)
3. Выбирается переход с вероятностью a[j]/sum(a)
4. Маркировка обновляется в соответствии с выбранным переходом
5. Процесс повторяется до достижения tmax или наступления deadlock

Возвращает: DataFrame с временем и маркировками
"""

function simulate_stochastic(
    net::PetriNet,
    u0::Vector{Float64},
    tmax::Float64;
    rates = ones(net.n_transitions),
    rng = Random.GLOBAL_RNG,
)
    u = copy(u0)
    t = 0.0
    times = [t]
    states = [copy(u)]
    
    while t < tmax
        # Вычисление скоростей переходов
        a = zeros(net.n_transitions)
        for j = 1:net.n_transitions
            rate = rates[j]
            prod = rate
            for i = 1:net.n_places
                if net.incidence[i, j] < 0
                    prod *= u[i] ^ (-net.incidence[i, j])
                end
            end
            a[j] = prod
        end
        
        a0 = sum(a)
        if a0 == 0
            break   # deadlock: нет активных переходов
        end
        
        # Время до следующего события
        dt = -log(rand(rng)) / a0
        
        # Выбор перехода
        r = rand(rng) * a0
        cumsum = 0.0
        chosen = 1
        for j = 1:net.n_transitions
            cumsum += a[j]
            if r <= cumsum
                chosen = j
                break
            end
        end
        
        # Обновление маркировки
        for i = 1:net.n_places
            u[i] += net.incidence[i, chosen]
        end
        
        t += dt
        if t <= tmax
            push!(times, t)
            push!(states, copy(u))
        end
    end
    
    # Формирование DataFrame
    df = DataFrame(time = times)
    for i = 1:net.n_places
        df[!, String(net.place_names[i])] = [s[i] for s in states]
    end
    return df
end

# Обнаружение deadlock

"""
detect_deadlock(df, net; tol)

Проверяет, находится ли сеть в состоянии deadlock в конце моделирования.

Deadlock наступает, когда ни один переход не может сработать.

Возвращает:
- true: обнаружен deadlock
- false: deadlock не обнаружен (есть активный переход)
"""

function detect_deadlock(df::DataFrame, net::PetriNet; tol = 1e-6)
    u_last = [df[end, String(net.place_names[i])] for i = 1:net.n_places]
    
    for j = 1:net.n_transitions
        can_fire = true
        for i = 1:net.n_places
            if net.incidence[i, j] < 0 && u_last[i] < -net.incidence[i, j] - tol
                can_fire = false
                break
            end
        end
        if can_fire
            return false   # найден активный переход
        end
    end
    return true            # активных переходов нет
end

# Визуализация

"""
plot_marking_evolution(df, N)

Создаёт графики эволюции маркировки для N философов.

Графики:
- Think: состояние "думает" для каждого философа
- Hungry: состояние "голоден" для каждого философа
- Eat: состояние "ест" для каждого философа
- Fork: состояние вилок для каждого философа

Возвращает: составной график из 4 панелей
"""

function plot_marking_evolution(df::DataFrame, N::Int)
    plots = []
    for group in ["Think", "Hungry", "Eat", "Fork"]
        p = plot(xlabel = "Time", ylabel = group, title = "$group states")
        for i = 1:N
            col = "$(group)_$i"
            if col in names(df)
                plot!(df.time, df[!, col], label = "$(group)_$i")
            end
        end
        push!(plots, p)
    end
    return plot(plots..., layout = (4, 1), size = (800, 1000))
end

end # module