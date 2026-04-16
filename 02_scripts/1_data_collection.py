# TFM: Automatización de la identificación de nuevas dianas farmacológicas mediante redes PPI
# Elaborado por: Nina Dudikova
# Fecha: feb 2026

# 1. DATA COLLECTION
# Objetivos: 
#   - 1.1 extraer los UniProt IDs de las proteínas diana conocidas a partir del archivo de DrugBank
#   - 1.2 convertir los UniProt IDs a gene names para poder consultar la base de datos STRING
#   - 1.3 obtener las PPI para cada target a parir de STRING
#   - 1.4 construir un archivo .csv con todas las interacciones obtenidas
#
# Tiempo de ejecución estimado: aproximadamente 45-50 minutos
#
# 0. Importación de paquetes
# Desde la terminal he instalado los paquetes requests y pandas
#   requests: hace llamadas a APIs externas
#   pandas: manejo de datos
#   zipfile: extracción de archivos comprimidos
#   os: gestión de archivos y rutas del sistema operativo
# Una vez instalados los paquetes, los cargo en memoria:

import zipfile
import os
import pandas as pd
import requests
import time

# 1.1 Extracción de UniProt IDs desde el archivo de DrugBank (tiempo de ejecución estimado: <1 segundo)
# Como paso previo, descargo manualmente el archivo drugbank_approved_target_polypeptide_sequences.fasta.zip de DrugBank.
# El zip contiene dos archivos: protein.fasta y gene.fasta.
# Solo necesito protein.fasta que contiene las secuencias de aminoácidos de todas las proteínas que son dianas farmacológicas conocidas.
# Cada proteína está identificada por su UniProt ID.
zip_path = "drugbank_approved_target_polypeptide_sequences.fasta.zip"

# Extraigo el archivo protein.fasta en la carpeta actual
with zipfile.ZipFile(zip_path, "r") as z:
    z.extract("protein.fasta")

# Una vez extraído, extraigo los UniProt IDs de todas las proteínas diana contenidas en el archivo FASTA de DrugBank.
uniprotIDs = [] 
with open("protein.fasta", "r") as f:
    for line in f:
        if line.startswith(">"):
            uniprotID = line.split("|")[1].split()[0].strip(";")
            uniprotIDs.append(uniprotID)

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open ("03_results/resultados.txt", "a") as f:
    f.write("Resultados de data collection: \n")
    f.write(f"UniProt IDs encontrados: {len(uniprotIDs)}\n")
print(f"UniProt IDs encontrados: {len(uniprotIDs)}") 


# 1.2 Conversión de UniProt IDs a gene names (tiempo de ejecución estimado: 35 segundos)
# La base de datos STRING no acepta UniProt IDs directamente, sino que trabaja con nombres de genes.
# Usando la API de UniProt convierto cada UniProt ID a su respectivo gene name. 
# Como espero que la lista de UniProt IDs sea larga, los IDs se procesan en grupos de 100, evitando hacer una llamada por cada ID. 

ID_to_gene = {}
batch_size = 100 

# Para cada grupo de 100 UniProt IDs, construyo la consulta, llamo a la API de UniProt, proceso la respuesta y guardo 
# los pares UniProt ID y gene name en el diccionario.
for i in range(0, len(uniprotIDs), batch_size):
    batch = uniprotIDs[i:i+batch_size] # Selección del grupo a procesar.
    query = " OR ".join(f"accession:{id}" for id in batch) # Construyo la consulta para la API uniendo todos los IDs del grupo con OR.

    # Llamo a la API de UniProt pidiendo los campos de accession y gene_names.
    response = requests.get(
        "https://rest.uniprot.org/uniprotkb/search",
        params={
            "query" : query,
            "fields": "accession,gene_names", 
            "format": "tsv",
            "size"  : batch_size,
        },
        timeout=15
    )

    # Limpio y divido la respuesta del API en líneas, elimino la primera línea que es la cabecera.
    lines = response.text.strip().split("\n")[1:] 
    
    # Recorro cada línea de la resuesta de la API para extraer el Uniprot ID y el gene name.
    for line in lines:
        # Divido cada línea por el tabulador para separar dos campos:
        #   part_lines[0] = UniProt ID
        #   parts_lines [1] = gene names
        parts_lines = line.split("\t")
        if len(parts_lines) >= 2: # Compruebo si la línea tiene dos campos
            u_ID = parts_lines[0].strip() # Extraigo el UniProtId de la primera columna.

            if parts_lines[1].strip(): # Compruebo que el segundo campo tenga contenido...
                gene_name = parts_lines[1].strip().split()[0] # ... si tiene contenido, extraigo el primer gene name.
            else:
                gene_name = None # ... si no tiene contenido, se le asigna None.
            if gene_name: # En el diccionario solo guardo lospares que tengan gene name válido
                ID_to_gene[u_ID] = gene_name
    time.sleep(0.3)

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open ("03_results/resultados.txt", "a") as f:
    f.write(f"Gene names obtenidos: {len(ID_to_gene)}\n")
print(f"Gene names obtenidos: {len(ID_to_gene)}")

# 1.3 Descarga de interacciones PPI desde STRING (tiempo de ejecución estimado: aprox. 45 minutos)
# Una vez convertidos los UniProt IDs a gene names, descargo las interacciones PPI de STRING.
# Consulto STRING para cada proteína target y STRING me devuelve sus proteínas interactoras.

all_interac = []
# Para cada gene name del diccionario ID_to_gene, consulto la API de STRING y descargo sus interacciones proteína-proteína.
for gene in ID_to_gene.values():
    # Uso try/except para que si STRING falla o reporta error,
    # el script no se rompa sino que continúe con la siguiente proteína.
    try:
        response = requests.get(
            "https://string-db.org/api/json/network",
            params={
                "identifiers"    : gene,  
                "species"        : 9606,  
                "required_score" : 700,   
                "limit"          : 50,    
                "caller_identity": "tfm", 
            },
            timeout=30 # Aumento el tiempo de espera para evitar fallos por conexión lenta.
        )
        data = response.json() # Convierto la respuesta JSON a una lista de diccionarios.
        
        # Compruebo que STRING no haya devuelto una respuesta vacía
        # ni un diccionario de error en vez de interacciones.
        # Sin esta comprobación el script fallaba con un KeyError.
        if not data or not isinstance(data, list):
            continue

        # Convierto la lista de diccionarios a un DataFrame de pandas,
        # quedándome con las 3 columnas de interés:
        #   preferredName_A = nombre de la proteína A
        #   preferredName_B = nombre de la proteína B, interactora
        #   score = confianza de la interacción
        df_interac_prot = pd.DataFrame(data)[["preferredName_A", "preferredName_B", "score"]]
        all_interac.append(df_interac_prot)
    except:
        continue
    time.sleep(0.5)

# 1.4 Construcción de un archivo .csv con todas las interacciones obtenidas
# Uno todos los dataframes de interacciones en uno solo, eliminando las filas repetidas
df_interac_all_prot = pd.concat(all_interac, ignore_index=True).drop_duplicates()

# Guardo el resultado en el archivo de texto y lo muestro en la terminal
with open ("03_results/resultados.txt", "a") as f:
    f.write(f"Total aristas: {len(df_interac_all_prot)}\n")
    f.write("-" * 40 + "\n")
print(f"Total aristas: {len(df_interac_all_prot)}")

# Guardo todas las interacciones en un archivo .csv
df_interac_all_prot.to_csv("edges.csv", index=False)

# Guardo la lista de gene names de los targets conocidos de DrugBank en un archivo CSV para usarla como labels positivos
pd.Series(list(ID_to_gene.values())).to_csv("targets.csv", index=False, header=False)

# Muevo todos los archivos generados al directorio 03_results
os.rename("edges.csv", "03_results/edges.csv")
os.rename("targets.csv", "03_results/targets.csv")

#Una vez procesados los archivos, los muevo al directorio 01_data
os.rename(zip_path, "01_data/drugbank_approved_target_polypeptide_sequences.fasta.zip")
os.rename("protein.fasta", "01_data/protein.fasta")