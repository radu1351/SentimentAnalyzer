import argparse
import os
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class FlancoSentimentAnalyzer:
    def __init__(self, reviews_file="flanco_reviews.csv", products_file="flanco_products.csv"):
        self.reviews_file = reviews_file
        self.products_file = products_file
        self.reviews_df = None
        self.products_df = None
        self.features_df = None

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

        try:
            self.romanian_stopwords = set(stopwords.words('romanian'))
        except:
            self.romanian_stopwords = set()
        self.english_stopwords = set(stopwords.words('english'))

        self.stop_words = self.romanian_stopwords.union(self.english_stopwords)

        additional_stopwords = [
            'si', 'în', 'la', 'cu', 'de', 'pe', 'pentru', 'este', 'sunt', 'fost',
            'dar', 'nici', 'foarte', 'mai', 'ca', 'sau', 'fie', 'iar', 'doar',
            'acest', 'aceasta', 'acestei', 'aceste', 'astfel', 'care', 'cand',
            'ce', 'cel', 'ceva', 'cum', 'ea', 'ei', 'el', 'eu', 'iti', 'va', 'vom'
        ]
        self.stop_words.update(additional_stopwords)

        self.stemmer = SnowballStemmer("romanian")

        self.feature_keywords = [
            'baterie', 'ecran', 'display', 'camera', 'procesor', 'memorie', 'ram',
            'stocare', 'ssd', 'hdd', 'greutate', 'design', 'sunet', 'difuzor',
            'autonomie', 'încărcare', 'performanță', 'viteză', 'preț', 'calitate',
            'touchpad', 'tastatură', 'încălzire', 'răcire', 'portabil', 'rezoluție',
            'software', 'sistem', 'windows', 'android', 'ios', 'conectivitate',
            'wifi', 'bluetooth', 'usb', 'hdmi', 'procesare', 'gaming', 'grafică',
            'gpu', 'cpu', 'ergonomie', 'preț', 'cost', 'valoare', 'bani'
        ]

        self.stemmed_features = [self.stemmer.stem(word) for word in self.feature_keywords]
        self.feature_keywords.extend(self.stemmed_features)

    def load_data(self):
        try:
            self.reviews_df = pd.read_csv(self.reviews_file)
            print(f"S-au încărcat {len(self.reviews_df)} review-uri din {self.reviews_file}")

            self.products_df = pd.read_csv(self.products_file)
            print(f"S-au încărcat {len(self.products_df)} produse din {self.products_file}")

            self._preprocess_data()

            return True
        except Exception as e:
            print(f"Eroare la încărcarea datelor: {str(e)}")
            return False

    def _preprocess_data(self):
        try:
            self.reviews_df['date'] = pd.to_datetime(self.reviews_df['date'], errors='coerce')
        except:
            print("Nu s-a putut converti coloana 'date' la format datetime.")

        self.reviews_df['rating_numeric'] = pd.to_numeric(self.reviews_df['rating'], errors='coerce')

        if 'price' in self.products_df.columns:
            self.products_df['price_clean'] = self.products_df['price'].apply(self._clean_price)

        self.reviews_df['text_clean'] = self.reviews_df['text'].apply(self._clean_text)
        self.reviews_df['text_processed'] = self.reviews_df['text_clean'].apply(self._preprocess_text)

    def _clean_price(self, price_str):
        if not isinstance(price_str, str):
            return None

        price_clean = re.sub(r'[^\d.,]', '', price_str).replace(',', '.')

        try:
            return float(price_clean)
        except:
            return None

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()

        text = text.replace('\n', ' ')

        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        text = re.sub(r'[^\w\s]', ' ', text)

        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _preprocess_text(self, text):
        if not isinstance(text, str) or not text:
            return ""

        tokens = word_tokenize(text)

        tokens = [token for token in tokens if token.lower() not in self.stop_words]

        tokens = [self.stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    def analyze_sentiment(self):
        if self.reviews_df is None:
            print("Nu s-au încărcat date. Rulați mai întâi metoda load_data().")
            return False

        self.reviews_df['sentiment_score'] = self.reviews_df['text_clean'].apply(
            lambda text: TextBlob(str(text)).sentiment.polarity if pd.notnull(text) else None
        )

        self.reviews_df['sentiment'] = self.reviews_df['sentiment_score'].apply(
            lambda score: 'Pozitiv' if score > 0.1 else ('Negativ' if score < -0.1 else 'Neutru')
        )

        sentiment_counts = self.reviews_df['sentiment'].value_counts()
        print("\nDistribuția sentimentului:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.reviews_df)) * 100
            print(f"  {sentiment}: {count} review-uri ({percentage:.1f}%)")

        return True

    def extract_features(self):
        if self.reviews_df is None:
            print("Nu s-au încărcat date. Rulați mai întâi metoda load_data().")
            return False

        feature_mentions = []

        for idx, row in self.reviews_df.iterrows():
            text = row['text_clean']
            if not isinstance(text, str) or not text:
                continue

            words = text.split()
            for feature in self.feature_keywords:
                if feature in words or f"{feature} " in text or f" {feature}" in text:
                    context = self._extract_context(text, feature)

                    feature_mentions.append({
                        'review_id': idx,
                        'product_name': row['product_name'],
                        'feature': feature,
                        'context': context,
                        'sentiment_score': row['sentiment_score'],
                        'sentiment': row['sentiment']
                    })

        if feature_mentions:
            self.features_df = pd.DataFrame(feature_mentions)

            self.feature_stats = self.features_df.groupby('feature').agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()

            self.feature_stats.columns = ['feature', 'avg_sentiment', 'std_sentiment', 'mention_count']

            self.feature_stats = self.feature_stats.sort_values('mention_count', ascending=False)

            print("\nTop 10 caracteristici menționate:")
            for i, (_, row) in enumerate(self.feature_stats.head(10).iterrows(), 1):
                sentiment_text = "pozitiv" if row['avg_sentiment'] > 0 else (
                    "negativ" if row['avg_sentiment'] < 0 else "neutru")
                print(
                    f"  {i}. {row['feature']} - {row['mention_count']} mențiuni, sentiment {sentiment_text} ({row['avg_sentiment']:.2f})")

            return True
        else:
            print("Nu s-au găsit caracteristici în recenzii.")
            return False

    def _extract_context(self, text, feature):
        words = text.split()
        feature_indices = [i for i, word in enumerate(words) if feature in word]

        if not feature_indices:
            return ""

        idx = feature_indices[0]
        start = max(0, idx - 5)
        end = min(len(words), idx + 6)

        return ' '.join(words[start:end])

    def generate_visualizations(self, output_dir="."):
        if self.reviews_df is None:
            print("Nu s-au încărcat date. Rulați mai întâi metoda load_data().")
            return False

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=self.reviews_df, x='sentiment',
                           order=['Pozitiv', 'Neutru', 'Negativ'],
                           palette={'Pozitiv': 'green', 'Neutru': 'gray', 'Negativ': 'red'})

        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')

        plt.title('Distribuția Sentimentului în Review-uri')
        plt.xlabel('Sentiment')
        plt.ylabel('Număr de Review-uri')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
        plt.close()

        self._generate_wordclouds(output_dir)

        valid_data = self.reviews_df.dropna(subset=['rating_numeric', 'sentiment'])

        sentiment_counts = valid_data['sentiment'].value_counts()

        if len(sentiment_counts) > 0 and not valid_data.empty:
            plt.figure(figsize=(10, 6))

            existing_categories = sentiment_counts.index.tolist()

            try:
                sns.boxplot(x='sentiment', y='rating_numeric', data=valid_data,
                            order=[cat for cat in ['Pozitiv', 'Neutru', 'Negativ'] if cat in existing_categories])
                plt.title('Relația între Rating și Sentiment')
                plt.xlabel('Sentiment')
                plt.ylabel('Rating')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'rating_vs_sentiment.png'))
            except Exception as e:
                print(f"Eroare la generarea boxplot-ului: {e}")

                plt.figure(figsize=(10, 6))
                sns.stripplot(x='sentiment', y='rating_numeric', data=valid_data,
                              order=[cat for cat in ['Pozitiv', 'Neutru', 'Negativ'] if cat in existing_categories],
                              jitter=True)
                plt.title('Relația între Rating și Sentiment (Scatterplot)')
                plt.xlabel('Sentiment')
                plt.ylabel('Rating')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'rating_vs_sentiment_scatter.png'))
        else:
            print("Nu există suficiente date pentru a genera graficul rating vs sentiment")

        plt.close()

        if hasattr(self, 'feature_stats') and not self.feature_stats.empty:
            top_features = self.feature_stats.head(15).copy()

            plt.figure(figsize=(12, 8))

            bars = plt.barh(top_features['feature'], top_features['avg_sentiment'])
            for i, bar in enumerate(bars):
                if top_features['avg_sentiment'].iloc[i] < 0:
                    bar.set_color('red')
                elif top_features['avg_sentiment'].iloc[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('gray')

            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Sentimentul Mediu pentru Caracteristicile Menționate')
            plt.xlabel('Sentiment (negativ ← → pozitiv)')
            plt.ylabel('Caracteristică')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_sentiment.png'))
            plt.close()

            plt.figure(figsize=(12, 8))
            sns.barplot(x='mention_count', y='feature', data=top_features, color='blue')
            plt.title('Frecvența Mențiunilor per Caracteristică')
            plt.xlabel('Număr de Mențiuni')
            plt.ylabel('Caracteristică')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_mentions.png'))
            plt.close()

        self._perform_topic_modeling(output_dir)

        return True

    def _generate_wordclouds(self, output_dir):
        positive_text = ' '.join(self.reviews_df[self.reviews_df['sentiment'] == 'Pozitiv']['text_processed'])
        if positive_text:
            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white',
                                  stopwords=self.stop_words,
                                  max_words=100).generate(positive_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Cuvinte Frecvente în Review-uri Pozitive')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'positive_wordcloud.png'))
            plt.close()

        negative_text = ' '.join(self.reviews_df[self.reviews_df['sentiment'] == 'Negativ']['text_processed'])
        if negative_text:
            wordcloud = WordCloud(width=800, height=400,
                                  background_color='white',
                                  stopwords=self.stop_words,
                                  max_words=100).generate(negative_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Cuvinte Frecvente în Review-uri Negative')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'negative_wordcloud.png'))
            plt.close()

    def _perform_topic_modeling(self, output_dir, num_topics=5, num_words=10):
        vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                     stop_words=list(self.stop_words))

        dtm = vectorizer.fit_transform(self.reviews_df['text_processed'].fillna(''))

        lda = LatentDirichletAllocation(n_components=num_topics,
                                        random_state=42,
                                        max_iter=10)
        lda.fit(dtm)

        feature_names = vectorizer.get_feature_names_out()

        with open(os.path.join(output_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Topicuri identificate în review-uri:\n\n")

            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-num_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                f.write(f"Topic #{topic_idx + 1}: {', '.join(top_words)}\n\n")

        doc_topics = lda.transform(dtm)
        self.reviews_df['dominant_topic'] = doc_topics.argmax(axis=1) + 1

        plt.figure(figsize=(10, 6))
        topic_counts = self.reviews_df['dominant_topic'].value_counts().sort_index()
        sns.barplot(x=topic_counts.index, y=topic_counts.values)
        plt.title('Distribuția Topicurilor în Review-uri')
        plt.xlabel('Număr Topic')
        plt.ylabel('Număr de Review-uri')
        plt.xticks(range(len(topic_counts)), [f'Topic {i}' for i in topic_counts.index])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_distribution.png'))
        plt.close()

        return topic_counts

    def cluster_reviews(self, output_dir=".", n_clusters=3):
        tfidf_vectorizer = TfidfVectorizer(max_features=1000,
                                           stop_words=list(self.stop_words))

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.reviews_df['text_processed'].fillna(''))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)

        self.reviews_df['cluster'] = clusters

        cluster_info = []

        for cluster_id in range(n_clusters):
            cluster_reviews = self.reviews_df[self.reviews_df['cluster'] == cluster_id]

            avg_sentiment = cluster_reviews['sentiment_score'].mean()

            cluster_text = ' '.join(cluster_reviews['text_processed'].fillna(''))
            word_freq = Counter(cluster_text.split())
            top_words = [word for word, _ in word_freq.most_common(10)]

            cluster_info.append({
                'cluster_id': cluster_id,
                'size': len(cluster_reviews),
                'avg_sentiment': avg_sentiment,
                'top_words': top_words
            })

        with open(os.path.join(output_dir, 'clusters.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Clustere identificate în review-uri:\n\n")

            for cluster in cluster_info:
                sentiment_text = "pozitiv" if cluster['avg_sentiment'] > 0 else (
                    "negativ" if cluster['avg_sentiment'] < 0 else "neutru")
                f.write(f"Cluster #{cluster['cluster_id'] + 1}:\n")
                f.write(f"  Număr de review-uri: {cluster['size']}\n")
                f.write(f"  Sentiment mediu: {sentiment_text} ({cluster['avg_sentiment']:.2f})\n")
                f.write(f"  Cuvinte cheie: {', '.join(cluster['top_words'])}\n\n")

        plt.figure(figsize=(12, 6))
        cluster_sentiment = self.reviews_df.groupby('cluster')['sentiment_score'].mean().sort_values()

        colors = ['red' if val < 0 else 'green' for val in cluster_sentiment.values]
        sns.barplot(x=cluster_sentiment.index, y=cluster_sentiment.values, palette=colors)

        plt.title('Sentimentul Mediu per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Sentiment Mediu')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(range(len(cluster_sentiment)), [f'Cluster {i + 1}' for i in cluster_sentiment.index])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_sentiment.png'))
        plt.close()

        return cluster_info

    def train_sentiment_classifier(self, output_dir="."):
        X = self.reviews_df['text_processed'].fillna('')

        y = self.reviews_df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        tfidf_vectorizer = TfidfVectorizer(max_features=1000,
                                           stop_words=list(self.stop_words))
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        classifier = MultinomialNB()
        classifier.fit(X_train_tfidf, y_train)

        y_pred = classifier.predict(X_test_tfidf)

        class_report = classification_report(y_test, y_pred, output_dict=True)

        with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("Raport de clasificare pentru sentimente:\n\n")
            f.write(f"Accuracy: {class_report['accuracy']:.2f}\n\n")

            f.write("Metrics per class:\n")
            for class_name in ['Pozitiv', 'Neutru', 'Negativ']:
                if class_name in class_report:
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {class_report[class_name]['precision']:.2f}\n")
                    f.write(f"    Recall: {class_report[class_name]['recall']:.2f}\n")
                    f.write(f"    F1-score: {class_report[class_name]['f1-score']:.2f}\n")
                    f.write(f"    Support: {class_report[class_name]['support']}\n\n")

        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classifier.classes_,
                    yticklabels=classifier.classes_)
        plt.title('Matricea de Confuzie pentru Clasificatorul de Sentiment')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        return classifier, tfidf_vectorizer, class_report

    def run_complete_analysis(self, output_dir="flanco_analysis"):
        os.makedirs(output_dir, exist_ok=True)

        print("Începem analiza completă a review-urilor Flanco...")

        if not self.load_data():
            print("Eroare la încărcarea datelor. Analiză oprită.")
            return False

        self.analyze_sentiment()

        self.extract_features()

        print("\nGenerăm vizualizări...")
        self.generate_visualizations(output_dir)

        print("\nRealizăm clusterizarea review-urilor...")
        self.cluster_reviews(output_dir)

        print("\nAntrenăm clasificatorul de sentiment...")
        self.train_sentiment_classifier(output_dir)

        self._generate_final_report(output_dir)

        print(f"\nAnaliza completă! Toate rezultatele au fost salvate în directorul '{output_dir}'.")

        return True

    def _generate_final_report(self, output_dir):
        with open(os.path.join(output_dir, 'raport_final.txt'), 'w', encoding='utf-8') as f:
            f.write("=== RAPORT DE ANALIZĂ PENTRU REVIEW-URI FLANCO ===\n\n")

            f.write("STATISTICI GENERALE\n")
            f.write("-------------------\n")
            f.write(f"Total produse analizate: {len(self.products_df)}\n")
            f.write(f"Total review-uri analizate: {len(self.reviews_df)}\n")

            sentiment_counts = self.reviews_df['sentiment'].value_counts()
            f.write("\nDISTRIBUȚIA SENTIMENTULUI\n")
            f.write("------------------------\n")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(self.reviews_df)) * 100
                f.write(f"  {sentiment}: {count} review-uri ({percentage:.1f}%)\n")

            avg_sentiment = self.reviews_df['sentiment_score'].mean()
            f.write(f"\nScorul mediu de sentiment: {avg_sentiment:.3f}\n")

            if 'rating_numeric' in self.reviews_df.columns:
                avg_rating = self.reviews_df['rating_numeric'].mean()
                f.write(f"Rating-ul mediu: {avg_rating:.2f} din 5\n")

            if hasattr(self, 'feature_stats') and not self.feature_stats.empty:
                f.write("\nCARACTERISTICI MENȚIONATE\n")
                f.write("-----------------------\n")
                f.write("Top 10 caracteristici după numărul de mențiuni:\n")

                for i, (_, row) in enumerate(self.feature_stats.head(10).iterrows(), 1):
                    sentiment_text = "pozitiv" if row['avg_sentiment'] > 0 else (
                        "negativ" if row['avg_sentiment'] < 0 else "neutru")
                    f.write(
                        f"  {i}. {row['feature']} - {row['mention_count']} mențiuni, sentiment {sentiment_text} ({row['avg_sentiment']:.2f})\n")

                f.write("\nTop 5 caracteristici cu sentiment pozitiv:\n")
                positive_features = self.feature_stats[self.feature_stats['mention_count'] >= 3].sort_values(
                    'avg_sentiment', ascending=False).head(5)
                for i, (_, row) in enumerate(positive_features.iterrows(), 1):
                    f.write(f"  {i}. {row['feature']} - {row['avg_sentiment']:.2f}, {row['mention_count']} mențiuni\n")

                f.write("\nTop 5 caracteristici cu sentiment negativ:\n")
                negative_features = self.feature_stats[self.feature_stats['mention_count'] >= 3].sort_values(
                    'avg_sentiment').head(5)
                for i, (_, row) in enumerate(negative_features.iterrows(), 1):
                    f.write(f"  {i}. {row['feature']} - {row['avg_sentiment']:.2f}, {row['mention_count']} mențiuni\n")

            f.write("\nANALIZA PE PRODUSE\n")
            f.write("------------------\n")

            product_counts = self.reviews_df['product_name'].value_counts().head(5)
            f.write("Top 5 produse după numărul de review-uri:\n")
            for i, (product, count) in enumerate(product_counts.items(), 1):
                f.write(f"  {i}. {product} - {count} review-uri\n")

            product_sentiment = self.reviews_df.groupby('product_name').agg({
                'sentiment_score': 'mean',
                'sentiment': 'count'
            }).rename(columns={'sentiment': 'count'})

            product_sentiment = product_sentiment[product_sentiment['count'] >= 3]

            f.write("\nTop 5 produse cu cel mai pozitiv sentiment (minim 3 review-uri):\n")
            top_positive = product_sentiment.sort_values('sentiment_score', ascending=False).head(5)

            for i, (product, row) in enumerate(top_positive.iterrows(), 1):
                f.write(f"  {i}. {product} - scor sentiment: {row['sentiment_score']:.2f}, {row['count']} review-uri\n")

            f.write("\nTop 5 produse cu cel mai negativ sentiment (minim 3 review-uri):\n")
            top_negative = product_sentiment.sort_values('sentiment_score').head(5)

            for i, (product, row) in enumerate(top_negative.iterrows(), 1):
                f.write(f"  {i}. {product} - scor sentiment: {row['sentiment_score']:.2f}, {row['count']} review-uri\n")

            f.write("\nTOPICURI IDENTIFICATE ÎN REVIEW-URI\n")
            f.write("-------------------------------\n")
            f.write("Consultați fișierul 'topics.txt' pentru detalii despre topicurile identificate.\n")

            f.write("\nCLUSTERE IDENTIFICATE\n")
            f.write("---------------------\n")
            f.write("Consultați fișierul 'clusters.txt' pentru detalii despre clusterele identificate.\n")

            f.write("\nCLASIFICAREA SENTIMENTULUI\n")
            f.write("-------------------------\n")
            f.write(
                "Consultați fișierul 'classification_report.txt' pentru detalii despre performanța clasificatorului.\n")

            f.write("\nFIȘIERE GENERATE\n")
            f.write("---------------\n")
            for filename in [
                'sentiment_distribution.png',
                'positive_wordcloud.png',
                'negative_wordcloud.png',
                'rating_vs_sentiment.png',
                'feature_sentiment.png',
                'feature_mentions.png',
                'topic_distribution.png',
                'cluster_sentiment.png',
                'confusion_matrix.png',
                'topics.txt',
                'clusters.txt',
                'classification_report.txt'
            ]:
                f.write(f"  - {filename}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Analizează sentimentul review-urilor de pe Flanco')

    parser.add_argument('--reviews-file', type=str, default="flanco_reviews.csv",
                        help='Calea către fișierul CSV cu review-uri (default: flanco_reviews.csv)')

    parser.add_argument('--products-file', type=str, default="flanco_products.csv",
                        help='Calea către fișierul CSV cu produse (default: flanco_products.csv)')

    parser.add_argument('--output-dir', type=str, default="flanco_analysis",
                        help='Directorul unde se vor salva rezultatele (default: flanco_analysis)')

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.reviews_file):
        print(f"Eroare: Fișierul {args.reviews_file} nu există!")
        print("Asigură-te că ai rulat extracția review-urilor înainte.")
        return 1

    analyzer = FlancoSentimentAnalyzer(
        reviews_file=args.reviews_file,
        products_file=args.products_file
    )

    analyzer.run_complete_analysis(output_dir=args.output_dir)

    return 0


if __name__ == "__main__":
    main()