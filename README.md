# Rent in Brazil

# A model for predicting the cost of rent in Brazil

This project was developed to determine the cost of rent in Brazil. The interactive interface is based on [**Streamlit**] (https://rent-mariisam.streamlit.app), which allows you to easily interact with the model and analyze the results.

<img src=“images/4.jpg” alt=“Image description” width=“500” height=“300”>

## Description.

**1. Project objective:** To create an analytical tool for researching the value of rents in Brazil, including various factors, including regions.

**2. Project tasks:**.

- Data analysis:\*\* to identify the key factors that influence the cost;
- Model building:\*\* using machine learning and statistical analyses to create a model that can predict rental values;
- User interface:\*\* develop an interactive interface that will allow users to enter new data, analyze the results and make predictions based on the model.

## Technologies.

The project was implemented using the following technologies:

- **Python**: the main programming language;
- **Docker Compose**: to simplify the process of deploying and managing the project in the Docker environment.

## Libraries

- **Pandas**: for data processing;
- Numpy: for numerical calculations;

- **Scikit-learn**: for building and evaluating machine learning models;
- **Matplotlib** and **Seaborn**: for data visualization;
- **Streamlit**: for creating an interactive interface;
- **Joblib**: for efficient serialization (saving) and loading of Python objects.

## Dataset

**The dataset used for this project has the following characteristics:**

- **https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset**
- format: `.csv`;
- contains the following key columns: `city`, `rent amount`, `bill_avg`, `parking spaces`, etc.

This dataset contains 10692 rental properties with 13 different characteristics:

- **city**: the city in which the property is located;
- **area**: area of the property;
- **rooms**: number of rooms;
- **bathroom**: number of bathrooms;
- **parking spaces**: number of parking spaces;
- **floor**: floor;
- **animals**: permission to stay with animals;
- **furniture**: furniture;
- **hoa (R$)** : homeowners association tax;
- **rent amount (R$)**: the amount of rent;
- **property tax**: Municipal property tax;
- **fire insurance (R$)**: the cost of fire insurance;
- **total (R$)**: the total sum of all values.

## The cost of renting an apartment depending on the number of rooms

![Вартість оренди в залежності від кількості кімнат](images/1.png)

## Rental housing costs depending on the city

![Rental housing costs depending on the city](images/2.png)

✅ **_In São Paulo, the average cost of rent is higher than in other cities._**
**_And this is not surprising, as it is the center of a large agglomeration with a population of 23 million, also one of the largest in the world._**

## Correlation analysis conclusions for rent amount:

![Correlation analysis conclusions for rent amount::](images/3.png)

# **_Correlation analysis conclusions for rent amount:_**

**_Strong positive relationship_**: Variables that are highly positively correlated with rent amount have a direct relationship. That is, when the value of these variables increases, the rent also increases.

- **fire insurance (0.987)**: Very strong correlation with rent amount. This is expected as the amount of insurance can be proportional to the rent.

- **bathrooms (0.666)**: Having more bathrooms is associated with higher rents. This indicates that living spaces with more amenities are more expensive.

- **parking spaces (0.574)**: Housing with parking spaces has higher rents, as it is often a sign of luxury or convenience.

- **rooms (0.537)**: A larger number of rooms is also associated with higher rents, which is consistent with the logic of larger areas.

- **City São Paulo (0.25)**: Location strongly influences higher rents.

**_Weak positive relationship_**:

- **area (0.178)**: Area has a moderate positive correlation, but is not a key factor. This may indicate that a larger area does not always mean significantly higher rents.

- **property tax (0.107)**: Property tax has a weak impact. It is probably taken into account by property owners, but is not a direct indicator of rents.

**_Almost neutral impact:_**

- **floor (0.071)**: Floor has a very weak impact on rents, which may depend on the city and the architecture of the buildings.

- **animal accept (0.06)**: Minor impacts on rent levels.

- **hoa (0.052)**: Small relationship with homeowners' association fees. This may only affect certain types of housing (e.g., apartments in condominiums).

**_Negative correlation_**: Variables with a negative correlation have an inverse relationship, i.e., when the value of the variable increases, the rent decreases.

- **animal_not_accept (-0.06)**: In premises where animals are not allowed, rents are slightly lower.

- **furniture_not furnished (-0.17)**: Show an inverse relationship, possibly due to tenant preferences.

Cities, for example, city_Belo Horizonte, city_Campinas: In cities with a negative correlation with rent amount, rents may be lower compared to the base cities (e.g. city_São Paulo).

## Модель

- the following models were tested in the project: **LinearRegression, Lasso, Ridge, ElasticNet, SVR, RandomForestRegressor**

- was used to select the best hyperparameters: GridSearchCV.

# Порівняння моделей:

| Model                           | MAE        | MSE          | RMSE        | R2       |
| ------------------------------- | ---------- | ------------ | ----------- | -------- |
| Linear Regression (Train)       | 296.182627 | 2.156508e+05 | 464.382169  | 0.981444 |
| Linear Regression (Test)        | 321.153450 | 6.703823e+05 | 818.768761  | 0.946516 |
| Lasso Regression (Train)        | 293.663125 | 2.247485e+05 | 474.076464  | 0.980661 |
| Lasso Regression (Test)         | 315.717036 | 3.939180e+05 | 627.628853  | 0.968573 |
| SVR (Train)                     | 284.258514 | 1.639049e+06 | 1280.253446 | 0.858968 |
| SVR (Test)                      | 291.741362 | 4.293574e+05 | 655.253668  | 0.965745 |
| Random Forest Regressor (Train) | 123.370415 | 5.525399e+04 | 235.061670  | 0.995246 |
| Random Forest Regressor (Test)  | 342.022606 | 9.401107e+05 | 969.593048  | 0.924996 |
| Ridge Regression (Train)        | 296.434150 | 2.156853e+05 | 464.419328  | 0.981441 |
| Ridge Regression (Test)         | 321.891060 | 6.876466e+05 | 829.244573  | 0.945138 |
| Elastic Net (Train)             | 294.810219 | 2.231826e+05 | 472.422034  | 0.980796 |
| Elastic Net (Test)              | 294.810219 | 2.231826e+05 | 472.422034  | 0.980796 |

# **_Comparisons and conclusions_**

❎ **_Best Model:_**

**Lasso Regression** offers the best balance between training and test performance, avoiding overfitting while maintaining strong explanatory power.

⭕ **_Worst Model:_**

Random Forest overfits the training data, with the highest test errors and reduced generalizability. It may require hyperparameter tuning to improve its performance.

## Run locally

1. **Clone the repository:**

```
git clone https://github.com/MariiaSam/Rent-in-Brazil.git
cd Rent-in-Brazil
```

2. **Set up the virtual environment with Poetry**

Set up project dependencies:

```
poetry install
```

To activate the virtual environment, run the command:

```
poetry shell
```

To add a dependency to a project, run the command:

```
poetry add <package_name>
```

To pull in existing dependencies:

```
poetry install
```

# Using

Run the Streamlit application with the command:

```
streamlit run app.py
```

<!-- # Docker

Цей проєкт також підтримує Docker-контейнеризацію, що дозволяє легко запускати додаток без необхідності налаштовувати середовище вручну.

## Запуск за допомогою Docker:

1. **Запуск проекту за допомогою Docker Compose**

У кореневій директорії проекту виконайте команду:

```
docker compose up
```

2. **Доступ до додатку:**

Після успішного запуску додаток буде доступний за адресою:

```
http://localhost:8501
```

3. **Зупинка проекту:**

Щоб зупинити проект, виконайте:

```
docker compose down
```

Ця команда зупинить усі сервіси та видалить створені контейнери. -->
