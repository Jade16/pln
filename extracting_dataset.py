import os
import zipfile
import glob
import pandas as pd

def extracting_dataset():
    # Encontrando o arquivo .zip
    zip_file = glob.glob("*.zip")
    zip_file = zip_file[0]

    # Checa se o arquivo zip existe
    if os.path.exists(os.path.basename(zip_file)):
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall()
                print(f"Extração bem-sucedida {zip_file}")
        except zipfile.BadZipFile:
            print(f"Erro: {zip_file} não é um arquivo zip.")
        except Exception as e:
            print(f"Ocorreu um erro durante a extração {zip_file}: {e}")
    else:
        print(f"Warning: {zip_file} não encontrado.")
   
    df = pd.read_csv(zip_file.replace(".zip", "")) 

    # Remove os valores NaN
    df.dropna(inplace=True)

    # Resetando o índice
    df.drop(columns=['original_index'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print(df)

    # Criando uma nova coluna com os valores de sentimento
    df['sentiment'] = df['rating'].apply(lambda x: 'negative' if x in [1, 2] else 'positive' if x in [4, 5] else 'neutral')
    df.head(10) 

    return df



