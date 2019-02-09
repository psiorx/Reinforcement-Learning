#pragma once

template<typename Iter, typename RandomGenerator>
inline Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
inline Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    Iter choice = select_randomly(start, end, gen);
    // std::cout << std::distance(start, choice) << "/" << std::distance(start, end) << std::endl;
    return choice;
}