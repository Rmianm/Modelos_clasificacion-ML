# Modelos de clasificacion

En los archivos explico la mayoría de las líneas de código, sin embargo aquí te explicaré de una manera más detallada.

Primero, cuando estés analizando el dataset te darás cuenta que algunas **variables son categóricas**, como el sexo de un paciente, y tú deseas cambiarla a una **variable numérica** de tipo Int, así que puedes elegir varios caminos para esto, hay algunos más eficientes y apropiados, esto también va de la mano del tipo de modelo y datos que estás implementando.

Una función que puedes utilizar es **LabelEncoder()** que se encuentra en el módulo `from sklearn import preprocessing`, en conjunto se usa con la librería pandas para codificar las variables, usemos de ejemplo una variable llamada sex que se encuentra en un dataset llamado train_df.

```sh
label_encoder = preprocessing.LabelEncoder()
encoder_sex = label_encoder.fit_transform(train_df['Sex'])
```

Si estás empezando, puedes hacerlo de una menera un poco más primitiva, aunque podrás tener inconvenientes si tiene muchas categorías
```sh
test_df['Sex'].replace('male',1,inplace=True)
test_df['Sex'].replace('female',0,inplace=True)
```

Otra manera de hacerlo es con **get_dummie** que a mi parecer es de las mejores. Es una función de Pandas que se utiliza para crear variables ficticias (también llamadas "variables dummy") a partir de variables categóricas.

`variable = pd.get_dummies(features)`

get_dummies utiliza **One-Hot Encoding**, es una técnica de transformación de variables categóricas en variables numéricas que consiste en crear una nueva columna para cada categoría de la variable original y asignar un valor binario (1 o 0) a cada fila de la nueva columna en función de si la fila pertenece o no a la categoría correspondiente, si pertenece le dará un valor de 1, sino le dará un 0.

LabelEncoder() al contrario de get_dummies puede ser un poco confuso para el modelo porque digamos que tenemos 3 categorías y a cada una le asigna un valor de 1, 2, 3. Esto le puede sonar al modelo que las variables tienen importancia en función del tamaño u orden de sus números.

Ahora bien, resulta que hay variables con valores nulos, esto podemos averiguarlo con un simple código `train_df.isnull().any()`, es importante saber que para implementar modelos de Machine learning las variables estén completas y no tengan valores faltantes por las siguientes razones.

1. La precisión del modelo se verá afectada: si hay muchos valores nulos en los datos, el modelo puede tener dificultades para encontrar patrones y relaciones entre las variables. Esto puede afectar negativamente la precisión del modelo y hacer que produzca predicciones inexactas.
2. La calidad de los datos se verá comprometida: los valores nulos pueden ser un indicador de que los datos tienen problemas de calidad o que no se han recopilado de manera adecuada
Entonces para dar solución a este problema puedes utilizar alternativas simples como utilizar el método **.fillna**: 

```sh
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
```

Se pueden llenar los datos con la media o la mediana, pero ten en cuenta que pueden haber outliers o valores atípicos, entonces la media podría verse afectada por estos y no representar adecuadamente la tendencia central de los datos, y si los valores núlos se completan de manera incorrecta estos pueden comprometer aún más la calidad de los datos y llevar a conclusiones erróneas.

