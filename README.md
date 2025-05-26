# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dari berbagai platform digital modern. Khususnya dalam industri hiburan seperti film, sistem rekomendasi berperan penting dalam membantu pengguna menemukan konten yang mungkin mereka sukai berdasarkan preferensi mereka sebelumnya. Dengan banjirnya pilihan film yang tersedia saat ini, pengguna sering kali kewalahan saat mencari film yang sesuai dengan selera mereka. Sistem rekomendasi membantu mengatasi masalah ini dengan menyarankan film-film yang kemungkinan besar akan disukai pengguna berdasarkan informasi yang sudah ada.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film yang dapat membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Dengan memanfaatkan dataset MovieLens, proyek ini akan mengimplementasikan dua pendekatan utama dalam sistem rekomendasi: Content-based Filtering dan Collaborative Filtering dengan Deep Learning menggunakan PyTorch.

Pengembangan sistem rekomendasi film penting karena beberapa alasan:

1. **Meningkatkan pengalaman pengguna**: Dengan menyarankan film yang relevan, pengguna dapat menemukan konten yang mereka sukai tanpa harus mencari secara manual.
2. **Meningkatkan engagement**: Rekomendasi yang akurat dapat meningkatkan waktu yang dihabiskan pengguna pada platform dan meningkatkan kepuasan.
3. **Mendorong penemuan konten**: Membantu pengguna menemukan film-film yang mungkin tidak mereka temukan sendiri, memperluas wawasan dan preferensi mereka.

Menurut penelitian oleh Davidson et al. (2010) dalam paper "The YouTube Video Recommendation System", sistem rekomendasi yang baik dapat meningkatkan engagement pengguna hingga 50% pada platform konten [1]. Selain itu, studi oleh Gomez-Uribe dan Hunt (2015) menunjukkan bahwa sistem rekomendasi Netflix berkontribusi lebih dari 80% terhadap konten yang ditonton pengguna [2].

## Business Understanding

### Problem Statements

Berikut adalah pernyataan masalah yang ingin diselesaikan dalam proyek ini:

1. Bagaimana cara merekomendasikan film yang sesuai dengan preferensi pengguna berdasarkan konten film (genre) yang mereka sukai sebelumnya?
2. Bagaimana cara merekomendasikan film kepada pengguna berdasarkan pola penilaian dari pengguna lain yang memiliki preferensi serupa?
3. Bagaimana mengukur keefektifan dari sistem rekomendasi yang dikembangkan?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan sistem rekomendasi berbasis konten (content-based) yang dapat menyarankan film dengan karakteristik serupa (genre) dari film yang disukai pengguna sebelumnya.
2. Membangun sistem rekomendasi kolaboratif (collaborative filtering) dengan Deep Learning menggunakan PyTorch yang dapat memprediksi film yang mungkin disukai pengguna berdasarkan preferensi pengguna lain yang serupa.
3. Mengevaluasi performa kedua pendekatan sistem rekomendasi dengan metrik yang tepat dan membandingkan kelebihan serta kekurangan masing-masing.

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, berikut adalah pendekatan yang akan diimplementasikan:

1. **Content-based Filtering**:
   - Menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstrak fitur genre dari film
   - Menghitung cosine similarity antara film-film untuk mengidentifikasi film yang memiliki kemiripan genre
   - Merekomendasikan film dengan genre serupa dari film yang disukai pengguna

2. **Collaborative Filtering dengan Deep Learning (PyTorch)**:
   - Mengembangkan model neural network untuk mempelajari pola tersembunyi dari interaksi pengguna-film
   - Menggunakan teknik embedding untuk merepresentasikan pengguna dan film dalam space fitur laten
   - Memprediksi rating yang mungkin diberikan pengguna untuk film yang belum mereka tonton
   - Merekomendasikan film dengan prediksi rating tertinggi

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/), yang merupakan dataset populer untuk penelitian sistem rekomendasi. Dataset ini berisi 100.000 rating dari 943 pengguna pada 1.682 film, dengan setiap pengguna memberikan rating pada setidaknya 20 film.

Dataset ini terdiri dari beberapa file, namun yang utama digunakan dalam proyek ini adalah:

1. **u.data**: Berisi data rating pengguna terhadap film
2. **u.item**: Berisi informasi mengenai film, termasuk judul dan genre
3. **u.user**: Berisi informasi demografis pengguna

Variabel-variabel pada dataset MovieLens 100K adalah sebagai berikut:

**Data Rating (u.data)**:
- user_id: ID unik pengguna (1 hingga 943)
- movie_id: ID unik film (1 hingga 1682)
- rating: Rating yang diberikan pengguna untuk film (integer 1-5)
- timestamp: Waktu rating diberikan

**Data Film (u.item)**:
- movie_id: ID unik film
- title: Judul film
- release_date: Tanggal rilis film
- IMDb_URL: URL IMDb film
- genre: 19 kolom boolean yang menunjukkan genre film (Action, Adventure, Animation, dll.)

**Data Pengguna (u.user)**:
- user_id: ID unik pengguna
- age: Usia pengguna
- gender: Jenis kelamin pengguna (M atau F)
- occupation: Pekerjaan pengguna
- zip_code: Kode pos pengguna

### Exploratory Data Analysis

Beberapa insight yang diperoleh dari exploratory data analysis:

#### Informasi Dataset

Berikut adalah informasi mengenai dataset yang digunakan:

1. **Dataset Ratings (u.data)**:
   - Jumlah data: 100.000 baris dan 4 kolom
   - Berisi informasi rating yang diberikan pengguna terhadap film

2. **Dataset Movies (u.item)**:
   - Jumlah data: 1.682 baris dan 24 kolom
   - Berisi informasi detail tentang film, termasuk judul dan genre

3. **Dataset Users (u.user)**:
   - Jumlah data: 943 baris dan 5 kolom
   - Berisi informasi demografis pengguna

#### Uraian Fitur Dataset

1. **Dataset Ratings (u.data)**:
   - `user_id`: ID unik pengguna (1 hingga 943)
   - `movie_id`: ID unik film (1 hingga 1682)
   - `rating`: Rating yang diberikan pengguna untuk film (integer 1-5)
   - `timestamp`: Waktu rating diberikan (dalam format UNIX timestamp)

2. **Dataset Movies (u.item)**:
   - `movie_id`: ID unik film
   - `title`: Judul film termasuk tahun rilis dalam tanda kurung
   - `release_date`: Tanggal rilis film dalam format DD-MMM-YYYY
   - `video_release_date`: Tanggal rilis video/DVD dalam format DD-MMM-YYYY (banyak nilai kosong karena pada saat data dikumpulkan, tidak semua film memiliki rilis video/DVD)
   - `IMDb_URL`: URL halaman IMDb untuk film tersebut
   - `unknown` hingga `Western`: 19 kolom boolean (0/1) yang menunjukkan genre film

3. **Dataset Users (u.user)**:
   - `user_id`: ID unik pengguna
   - `age`: Usia pengguna
   - `gender`: Jenis kelamin pengguna (M atau F)
   - `occupation`: Pekerjaan pengguna
   - `zip_code`: Kode pos tempat tinggal pengguna

#### Analisis Data Eksploratif

1. **Pemeriksaan Nilai Hilang dan Data Duplikat**
   Hasil analisis menunjukkan bahwa dataset *ratings* dan *users* tidak memiliki nilai yang hilang. Sementara itu, pada dataset *movies*, ditemukan sejumlah nilai yang hilang, yaitu: 1.682 nilai pada kolom `video_release_date`, 1 nilai pada `release_date`, dan 3 nilai pada `IMDb_URL`. Namun, karena ketiga kolom tersebut tidak digunakan dalam proses pemodelan sistem rekomendasi, kekurangan data ini tidak berdampak signifikan terhadap analisis yang dilakukan. Selain itu, hasil pemeriksaan terhadap data duplikat mengonfirmasi bahwa tidak terdapat data yang terduplikasi pada ketiga dataset. Dengan demikian, kita dapat menyimpulkan bahwa data telah bersih dan siap digunakan untuk tahap analisis dan pemodelan selanjutnya.

2. **Distribusi Rating**:
   Dataset memiliki distribusi rating yang cenderung positif, dengan mayoritas rating adalah 4 dari skala 1-5. Hal ini menunjukkan bahwa pengguna cenderung memberikan rating pada film yang mereka sukai.

3. **Distribusi Genre**:
   Genre "Drama" adalah genre yang paling umum dalam dataset, diikuti oleh "Comedy" dan "Action". Genre yang paling sedikit adalah "Film-Noir" dan "Documentary".

4. **Aktivitas Pengguna**:
   Terdapat variasi yang signifikan dalam jumlah rating yang diberikan oleh pengguna. Beberapa pengguna sangat aktif dan memberikan rating pada ratusan film, sementara yang lain hanya memberikan rating pada jumlah film minimum (20).

5. **Popularitas Film**:
   Film-film tertentu menerima jauh lebih banyak rating dibandingkan yang lain, menunjukkan adanya film-film populer yang ditonton oleh banyak orang. Film "Star Wars (1977)" adalah salah satu film yang paling banyak dirating.

## Data Preparation

Beberapa tahapan data preparation yang dilakukan dalam proyek ini adalah:

### 1. Pemrosesan Genre Film

Untuk Content-based Filtering, genre film yang awalnya tersebar dalam 19 kolom boolean diubah menjadi satu string yang berisi semua genre film. Hal ini memudahkan proses perhitungan kemiripan antar film berdasarkan genre untuk pendekatan Content-Based Filtering.

```python
def create_genre_string(row):
    genres = []
    for genre in genre_columns:
        if row[genre] == 1:
            genres.append(genre)
    return ' '.join(genres)

movies['genres'] = movies.apply(create_genre_string, axis=1)
```

Sebagai contoh, film yang sebelumnya memiliki nilai 1 pada kolom "Action" dan "Adventure" akan memiliki string genre "Action Adventure". Transformasi ini diperlukan untuk tahapan selanjutnya yaitu ekstraksi fitur dengan TF-IDF.

### 2. Ekstraksi Fitur dengan TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode statistik yang digunakan untuk mengevaluasi pentingnya sebuah kata dalam dokumen dari kumpulan dokumen. Dalam konteks sistem rekomendasi film, kita menggunakan TF-IDF untuk mengekstrak fitur dari string genre film dan mengubahnya menjadi vektor numerik yang merepresentasikan karakteristik film. 

Proses ini terdiri dari dua komponen:
- **Term Frequency (TF)**: Mengukur seberapa sering sebuah kata muncul dalam dokumen. Dalam kasus ini, menghitung frekuensi genre dalam string genre film.
- **Inverse Document Frequency (IDF)**: Mengukur pentingnya sebuah kata. Kata yang jarang muncul di seluruh dokumen memiliki nilai IDF tinggi. Dalam konteks film, genre yang jarang (seperti "Film-Noir") akan memiliki bobot lebih tinggi dibanding genre yang umum (seperti "Drama").

Tahap ini akan dilakukan pada bagian modeling untuk Content-Based Filtering.

### 3. Normalisasi Rating

Untuk Collaborative Filtering, rating yang awalnya dalam skala 1-5 dinormalisasi menjadi skala 0-1. Hal ini dilakukan untuk memudahkan proses pelatihan model neural network.

```python
ratings['normalized_rating'] = ratings['rating'] / 5.0
```

Normalisasi rating diperlukan karena fungsi aktivasi sigmoid pada model neural network menghasilkan output dalam rentang 0-1.

### 4. Pembagian Data Training dan Testing

Data rating dibagi menjadi training set (80%) dan testing set (20%) untuk melatih dan mengevaluasi model Collaborative Filtering menggunakan metode train_test_split dari scikit-learn.

```python
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
```

Pembagian data ini penting untuk mengevaluasi kinerja model pada data yang belum pernah dilihat model sebelumnya, sehingga dapat menilai kemampuan generalisasi model.

### 5. Pembuatan Dataset dan DataLoader PyTorch

Untuk melatih model neural network, data rating dikonversi menjadi format yang sesuai dengan PyTorch melalui pembuatan kelas dataset kustom dan dataloader. Hal ini memungkinkan batch processing dan meningkatkan efisiensi proses training.

```python
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        user_id = self.df.iloc[idx]['user_id'] - 1
        movie_id = self.df.iloc[idx]['movie_id'] - 1
        rating = self.df.iloc[idx]['normalized_rating']
        
        return {'user_id': torch.tensor(user_id, dtype=torch.long),
                'movie_id': torch.tensor(movie_id, dtype=torch.long),
                'rating': torch.tensor(rating, dtype=torch.float)}
```

Tahapan ini diperlukan untuk memfasilitasi proses training yang efisien dengan batching dan shuffling data.

## Modeling

Dalam proyek ini, dua pendekatan sistem rekomendasi telah diimplementasikan:

### 1. Content-Based Filtering

Content-based filtering merekomendasikan film berdasarkan kemiripan konten atau atribut item dengan item yang disukai pengguna sebelumnya. Pendekatan ini menganalisis fitur film (dalam hal ini genre) dan mencari film lain dengan karakteristik serupa.

**Cara Kerja Content-Based Filtering:**
1. Mengekstraksi fitur dari item (dalam hal ini genre film)
2. Membuat profil item berdasarkan fitur-fiturnya
3. Menghitung kesamaan antar item menggunakan metrik jarak/similarity
4. Merekomendasikan item yang paling mirip dengan item referensi

**Tahapan implementasi**:
1. Menggunakan TF-IDF Vectorizer untuk mengkonversi string genre menjadi vektor numerik
2. Menghitung cosine similarity antar film berdasarkan vektor TF-IDF
3. Membuat fungsi rekomendasi yang mengembalikan top-N film yang paling mirip dengan film yang ditentukan

```python
# Menggunakan TF-IDF untuk mengkonversi data genre menjadi vektor
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Menghitung kemiripan kosinus antar film
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

**Contoh rekomendasi untuk "Star Wars (1977)"**:

| No | ID Film | Judul Film | Genre | Similarity Score |
|----|----|-----------|-------|------------------|
| 1 | 180 | Return of the Jedi (1983) | Action Adventure Romance Sci-Fi War | 1.000000 |
| 2 | 171 | Empire Strikes Back, The (1980) | Action Adventure Drama Romance Sci-Fi War | 0.978268 |
| 3 | 270 | Starship Troopers (1997) | Action Adventure Sci-Fi War | 0.941994 |
|4 | 120 | Independence Day (ID4) (1996) | Action Sci-Fi War | 0.850581 |
| 5 | 234 | Mars Attacks! (1996) | Action Comedy Sci-Fi War | 0.815144 |
| 6 | 61 | Stargate (1994) | Action Adventure Sci-Fi | 0.811689 |
| 7 | 81 | Jurassic Park (1993) | Action Adventure Sci-Fi | 0.811689 |
| 8 | 221 | Star Trek: First Contact (1996) |	Action Adventure Sci-Fi | 0.811689 |
| 9 | 226 | Star Trek VI: The Undiscovered Country (1991) | Action Adventure Sci-Fi | 0.811689 |
| 10 | 227	 | Star Trek: The Wrath of Khan (1982) | Action Adventure Sci-Fi | 0.811689

Film referensi "Star Wars (1977)" memiliki genre: Action Adventure Sci-Fi. Dari hasil di atas, kita dapat melihat bahwa sistem merekomendasikan film-film dengan genre serupa, terutama film sci-fi dan action. Dua film teratas adalah sekuel dari film referensi dengan skor kemiripan yang sangat tinggi.

**Contoh rekomendasi untuk "Toy Story (1995)"**:

| No | ID Film | Judul Film | Genre | Similarity Score |
|----|----|-----------|-------|------------------|
| 1 | 421 | Aladdin and the King of Thieves (1996) | Animation Children Comedy | 1.000000 |
| 2 | 101 | Aristocats, The (1970) | Animation Children |	0.936967 |
| 3 | 403 | Pinocchio (1940) | Animation Children | 0.936967 |
| 4 | 624 | Sword in the Stone, The (1963) | Animation Children | 0.936967 |
| 5 | 945 | Fox and the Hound, The (1981) |Animation Children | 0.936967 |
| 6 | 968	| Winnie the Pooh and the Blustery Day (1968) | Animation Children | 0.936967 |
| 7 | 1065 | Balto (1995) | Animation Children | 0.936967 |
| 8 | 1077 | Oliver & Company (1988) | Animation Children | 0.936967 |
| 9 | 1408 | Swan Princess, The (1994) | Animation Children | 0.936967 |
| 10 | 1411 | Land Before Time III: The Time of the Great Giving (1995) |	Animation Children | 0.936967

Film referensi "Toy Story (1995)" memiliki genre: Animation Children's Comedy. Dapat dilihat bahwa rekomendasi didominasi oleh film animasi anak-anak, menunjukkan kemampuan sistem dalam mengidentifikasi film-film dengan karakteristik serupa.

**Kelebihan**:
- Tidak memerlukan data dari pengguna lain
- Dapat merekomendasikan item baru atau tidak populer
- Dapat memberikan penjelasan mengapa item direkomendasikan (transparansi)

**Kekurangan**:
- Terbatas pada fitur yang tersedia (hanya genre dalam kasus ini)
- Tidak dapat mempelajari preferensi tersembunyi pengguna
- Cenderung memberikan rekomendasi yang mirip saja (kurang eksplorasi)

### 2. Collaborative Filtering dengan Deep Learning (PyTorch)

Collaborative Filtering merekomendasikan item berdasarkan preferensi pengguna lain yang memiliki pola perilaku serupa. Berbeda dengan content-based filtering yang fokus pada fitur item, pendekatan ini beroperasi pada matriks interaksi user-item.

**Cara Kerja Collaborative Filtering dengan Neural Network:**
1. Mempelajari representasi tersembunyi (latent representation) dari pengguna dan item dalam ruang fitur dimensi rendah
2. Menggunakan embedding layers untuk memetakan ID user dan ID item ke vektor fitur laten
3. Menemukan pola tersembunyi dari interaksi user-item menggunakan deep neural network
4. Memprediksi rating atau preferensi pengguna untuk item yang belum pernah mereka interaksikan

Dalam implementasi ini, kita menggunakan Neural Collaborative Filtering (NCF), sebuah framework yang menggabungkan deep learning dengan collaborative filtering tradisional.

**Arsitektur Model**:
- Embedding layer untuk pengguna dan film
- Multi-layer perceptron (MLP) dengan aktivasi ReLU untuk memodelkan interaksi non-linear
- Output layer dengan aktivasi sigmoid untuk memprediksi rating (0-1)

```python
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=50, hidden_layers=[100, 50]):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_size = embedding_size * 2
        
        for i, hidden_size in enumerate(hidden_layers):
            self.fc_layers.append(nn.Linear(input_size, hidden_size))
            self.fc_layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
```

**Proses pelatihan**:
- Loss function: Mean Squared Error
- Optimizer: Adam dengan learning rate 0.001
- Batch size: 64
- Epochs: 10

**Contoh rekomendasi untuk User ID 5**:

| No | ID Film | Judul Film | Genre | Predicted Rating |
|----|----|-----------|-------|-----------------|
| 1 | 850 | Perfect Candidate, A (1996) | Documentary | 4.769566 |
| 2 | 1449 | Pather Panchali (1955) | Drama | 4.745034 |
| 3 | 114 | Wallace & Gromit: The Best of Aardman Animations (1996) | Animation | 4.705163 |
| 4 | 896 | Sweet Hereafter, The (1997) | Drama | 4.700777 |
| 5 | 867 | Whole Wide World, The (1996) | Drama | 4.678691 |
| 6 | 1500 | Santa with Muscles (1996) | Comedy | 4.665243 |
| 7 | 175 | Brazil (1985) | Sci-Fi | 4.642331 |
| 8 | 1174 | Caught (1996) | Drama Thriller | 4.624546 |
| 9 | 56 | Pulp Fiction (1994) | Crime Drama | 4.617467 |
| 10 | 1175 | Hugo Pool (1997) | Romance | 4.605856

Dari hasil di atas, kita melihat bahwa model collaborative filtering memprediksi preferensi film untuk User ID 5. Rekomendasi didominasi oleh film drama klasik dengan rating yang tinggi.

**Contoh rekomendasi untuk User ID 100**:

| No | ID Film | Judul Film | Genre | Predicted Rating |
|----|----|-----------|-------|-----------------|
| 1 | 318 | Schindler's List (1993) | Drama War | 4.818922
| 2 | 1293 | Star Kid (1997) | Adventure Children Fantasy Sci-Fi | 4.812152
| 3 | 1612 | Leading Man, The (1996) | Romance | 4.779630 |
| 4 | 64 | Shawshank Redemption, The (1994) | Drama | 4.776513 |
| 5 | 963 | Some Folks Call It a Sling Blade (1993) | Drama Thriller | 4.743431 |
| 6 | 1189 | Prefontaine (1997) | Drama | 4.733603 |
| 7 | 1449 | Pather Panchali (1955) | Drama | 4.708386 |
| 8 | 1251 | A Chef in Love (1996) | Comedy | 4.699285 |
| 9 | 50 | Star Wars (1977) | Action Adventure Romance Sci-Fi War | 4.696585 |
| 10 | 1500 | Santa with Muscles (1996) | Comedy | 4.689219 |

Dari hasil di atas, kita melihat bahwa model collaborative filtering memprediksi preferensi film untuk User ID 100. Rekomendasi didominasi oleh film Drama dengan rating yang tinggi

Model collaborative filtering dapat menghasilkan rekomendasi yang berbeda untuk pengguna yang berbeda, menunjukkan kemampuannya dalam memberikan rekomendasi yang dipersonalisasi berdasarkan pola rating setiap pengguna.

**Kelebihan**:
- Dapat merekomendasikan item di luar profil pengguna berdasarkan kesamaan dengan pengguna lain
- Tidak memerlukan data konten untuk memberikan rekomendasi
- Dapat memberikan rekomendasi yang lebih bervariasi

**Kekurangan**:
- Memerlukan data rating dari banyak pengguna (cold start problem)
- Tidak bisa merekomendasikan item baru yang belum pernah dirating (item cold start)
- Dapat memiliki masalah sparsity ketika dataset ratings jarang dan kecil

## Evaluation

Dalam proyek ini, beberapa metrik evaluasi digunakan untuk menilai performa sistem rekomendasi:

### 1. Metrik untuk Content-Based Filtering

**Average Precision**: Mengukur seberapa relevan film yang direkomendasikan berdasarkan kemiripan genre dengan film yang menjadi referensi.

Formula:
$\text{Precision} = \frac{\text{Jumlah genre yang sama}}{\text{Jumlah total genre film referensi}}$

**Coverage**: Mengukur seberapa bervariasi genre yang direkomendasikan.

Formula:
$\text{Coverage} = \frac{\text{Jumlah genre unik dalam rekomendasi}}{\text{Jumlah total genre yang tersedia}}$

**Hasil evaluasi content-based filtering**:
- Film "Star Wars (1977)": Average Precision = 0.7000, Coverage = 0.3684
- Film "Toy Story (1995)": Average Precision = 0.7000, Coverage = 0.1579
- Film "Pulp Fiction (1994)": Average Precision = 1.0000, Coverage = 0.1053

### 2. Metrik untuk Collaborative Filtering

**RMSE (Root Mean Squared Error)**: Mengukur rata-rata kuadrat selisih antara rating prediksi dengan rating aktual.

Formula:
$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

dimana $y_i$ adalah rating aktual dan $\hat{y}_i$ adalah rating prediksi.

**MAE (Mean Absolute Error)**: Mengukur rata-rata nilai absolut dari selisih antara rating prediksi dengan rating aktual.

Formula:
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

**Accuracy**: Proporsi prediksi yang mendekati rating aktual (dengan toleransi tertentu).

Formula:
$\text{Accuracy} = \frac{\text{Jumlah prediksi yang akurat}}{\text{Total prediksi}}$

**Hasil evaluasi collaborative filtering**:
- RMSE: 0.9699
- MAE: 0.7513
- Accuracy: 0.4252

### 3. Perbandingan Pendekatan

Kedua pendekatan sistem rekomendasi memiliki kekuatan dan kelemahan masing-masing:

**Content-Based Filtering** baik dalam merekomendasikan item berdasarkan kemiripan konten dan tidak memerlukan data dari pengguna lain. Namun, terbatas pada fitur yang tersedia dan cenderung memberikan rekomendasi yang kurang bervariasi.

**Collaborative Filtering** dapat menemukan pola tersembunyi dari interaksi pengguna-item dan memberikan rekomendasi yang lebih bervariasi. Namun, memiliki masalah cold start dan tidak bisa merekomendasikan item baru.

Hasil evaluasi menunjukkan bahwa collaborative filtering dengan deep learning memberikan performa dengan accuracy 42.52% dan RMSE 0.9699 (dalam skala 1-5). Nilai RMSE mendekati 1 mengindikasikan bahwa model memiliki error prediksi rating sekitar 1 poin pada skala 1-5. Performa ini menunjukkan sedikit peningkatan dibandingkan pengujian sebelumnya (RMSE 0.9763, accuracy 42.36%), yang kemungkinan disebabkan oleh optimasi hyperparameter dan manajemen seed yang lebih baik untuk stabilitas training. Meskipun peningkatan relatif kecil, hal ini menunjukkan bahwa kualitas training neural network sangat dipengaruhi oleh inisialisasi dan stabilitas proses pelatihan. Accuracy yang masih di bawah 50% menunjukkan bahwa model masih memiliki ruang untuk peningkatan, terutama dalam menangani sparsitas data, kompleksitas pola preferensi pengguna, atau mempertimbangkan arsitektur model yang lebih canggih. Content-based filtering tetap efektif dalam merekomendasikan film dengan genre serupa, dengan average precision sekitar 60-70%.

Untuk sistem rekomendasi yang optimal, pendekatan hybrid yang menggabungkan kekuatan kedua metode dapat menjadi solusi yang lebih baik.

## Referensi

[1] Davidson, J., Liebald, B., Liu, J., Nandy, P., Van Vleet, T., Gargi, U., ... & Sampath, D. (2010, September). The YouTube video recommendation system. In Proceedings of the fourth ACM conference on Recommender systems (pp. 293-296).

[2] Gomez-Uribe, C. A., & Hunt, N. (2015). The netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1-19.
