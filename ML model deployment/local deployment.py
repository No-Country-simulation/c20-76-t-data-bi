import joblib
import pandas as pd

#Cargamos el Modelo
model = joblib.load('lgb_model.pkl')

# Obtenemos la informacion del usuario
def get_user_input():
    #definimos los nombres de nuestras caracteristicas
    user_data = {}

    value = input(f'Enter Value for Category:')
    user_data['Category'] = value
    num_feature = ['Free','Price','Size','Minimum Android','Ad Supported']

    # Obtenemos la informacion
    for feature in num_feature:
        value = input(f'Enter Value for {feature}: ')
        user_data[feature] = [float(value)]

    
    return user_data

def main():
    targets = ['Rating','Installs']
    #obtenemos la informacion del usuarioEducation                  
    data = get_user_input()
    # guardamos la informacion en un data set
    df = pd.DataFrame(data)
    df['Category'] = df['Category'].astype('category')
    #hacemos nuestras predicciones
    prediction = model.predict(df)
    # imprimimos los resultados
    for i,value in enumerate(prediction[0]):
        formated_value = "{:,.2f}".format(value)
        print(f'{targets[i]}: {formated_value}')

if __name__ == '__main__':
    main()



