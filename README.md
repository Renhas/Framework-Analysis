Сравнительный анализ фреймворков для поиска гиперпараметров моделей машинного обучения:

* [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
* [Optuna](https://optuna.org/)
* [Hyperopt](https://hyperopt.github.io/hyperopt/)
* [Bayesian Optimization](https://github.com/bayesian-optimization/BayesianOptimization)
* [Talos](https://autonomio.github.io/talos/#/)

## `notebooks`
Основная папка, содержащая файлы Jupiter Nootebook, для:
* Анализа датасета (`Dataset`)
* Сбора данных по каждому фреймворку (`Frameworks`)
* Анализа результатов (`Analysis`)

## `Results`
Папка с результатами по всем фреймворкам

## `data`
Папка с промежуточными данными из датасета и с финальным, обработанным датасетов

## `src`
Папка с основными модулями:
* `loaders` - модуль с объектами, реализующими сохранение и загрузку результатов исследований. Содержит фабрики (`loaders_factory.py`) для удобного создания необходимых объектов.
* `managers` - модуль с менеджерами исследований, осуществляющих непосредственно исследование конкретных фреймворков. Содержит builder'ы (`managers_builders.py`) для удобного создания менеджеров.

## `tests`
Папка содержит тесты для модулей из `src`
