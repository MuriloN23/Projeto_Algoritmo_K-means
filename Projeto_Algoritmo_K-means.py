
# Caminhos relativos para os arquivos
caminho_features = "./UCI HAR Dataset/UCI HAR Dataset/features.txt"
caminho_x_train = "./UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt"
caminho_y_train = "./UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"
caminho_x_test = "./UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt"
caminho_y_test = "./UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt"

# Carregar os dados do dataset
features = pd.read_csv(caminho_features, sep='\s+', header=None)
x_train = pd.read_csv(caminho_x_train, sep='\s+', header=None)
y_train = pd.read_csv(caminho_y_train, sep='\s+', header=None)
x_test = pd.read_csv(caminho_x_test, sep='\s+', header=None)
y_test = pd.read_csv(caminho_y_test, sep='\s+', header=None)

