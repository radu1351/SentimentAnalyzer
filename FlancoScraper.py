import argparse
import csv
import os
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

class FlancoReviewExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        self.base_url = 'https://www.flanco.ro'
        self.all_reviews = []
        self.all_products = []

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')

        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"Eroare la initializarea browser-ului: {str(e)}")
            print("Asigura-te ca ai instalat Chrome si ChromeDriver!")
            exit(1)

    def accept_cookies(self):
        try:
            cookie_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#cc-wrapper .x-agree"))
            )
            self.driver.execute_script("arguments[0].click();", cookie_button)
            time.sleep(1)
            return True
        except (TimeoutException, NoSuchElementException):
            return False

    def get_product_urls(self, category_url, max_pages=2):
        product_urls = []

        for page in range(1, max_pages + 1):
            page_url = f"{category_url}?p={page}"

            try:
                response = requests.get(page_url, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                products = soup.select('.product-item')

                for product in products:
                    product_link = product.select_one('.product-item-link')
                    if product_link and product_link.has_attr('href'):
                        product_url = product_link['href']
                        if not product_url.startswith('http'):
                            product_url = self.base_url + product_url

                        product_urls.append(product_url)

                        product_name = product_link.get_text(strip=True) if product_link else "Nedisponibil"
                        price_elem = product.select_one('.price-final_price .price')
                        price = price_elem.get_text(strip=True) if price_elem else "Nedisponibil"
                        review_count_elem = product.select_one('.reviews-actions .action')
                        review_count = review_count_elem.get_text(strip=True).split()[0] if review_count_elem else "0"

                        self.all_products.append({
                            'name': product_name,
                            'url': product_url,
                            'price': price,
                            'review_count': review_count
                        })

                time.sleep(1)
            except Exception as e:
                print(f"Eroare la colectarea produselor de pe pagina {page}: {str(e)}")

        return product_urls

    def extract_reviews_with_selenium(self, product_url):
        product_reviews = []

        try:
            self.driver.get(product_url)

            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.product-info-main, .page-title'))
            )

            self.accept_cookies()

            product_name = "Nedisponibil"
            try:
                product_name_elem = self.driver.find_element(By.CSS_SELECTOR, '.page-title span')
                product_name = product_name_elem.text.strip()
            except NoSuchElementException:
                try:
                    product_name_elem = self.driver.find_element(By.CSS_SELECTOR, 'h1')
                    product_name = product_name_elem.text.strip()
                except NoSuchElementException:
                    pass

            try:
                rating_element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '.rating'))
                )

                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", rating_element)
                time.sleep(1)

                self.driver.execute_script("arguments[0].click();", rating_element)
                time.sleep(3)

            except TimeoutException:
                return []

            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.review-item'))
                )
            except TimeoutException:
                return []

            review_items = self.driver.find_elements(By.CSS_SELECTOR, '.review-item')

            for i, review in enumerate(review_items):
                try:
                    author = "Anonim"
                    try:
                        author_elem = review.find_element(By.CSS_SELECTOR, '.review-content__name')
                        author = author_elem.text.strip()
                    except NoSuchElementException:
                        pass

                    review_date = "Nedisponibil"
                    try:
                        date_elem = review.find_element(By.CSS_SELECTOR, '.review-date')
                        review_date = date_elem.text.strip()
                    except NoSuchElementException:
                        pass

                    rating = "0"
                    try:
                        rating_elem = review.find_element(By.CSS_SELECTOR, '.rating-result > span > span')
                        rating_text = rating_elem.text.strip()
                        digits = re.findall(r'\d+', rating_text)
                        if digits:
                            rating = digits[0]
                    except NoSuchElementException:
                        pass

                    text = ""
                    try:
                        text_elem = review.find_element(By.CSS_SELECTOR, '.review-item > div > p')
                        text = text_elem.text.strip()
                    except NoSuchElementException:
                        try:
                            text_elem = review.find_element(By.CSS_SELECTOR, 'p')
                            text = text_elem.text.strip()
                        except NoSuchElementException:
                            pass

                    title = ""
                    try:
                        title_elem = review.find_element(By.CSS_SELECTOR, '.review-title')
                        title = title_elem.text.strip()
                    except NoSuchElementException:
                        pass

                    if title and text:
                        full_text = f"{title}\n{text}"
                    else:
                        full_text = text or title

                    if full_text:
                        review_data = {
                            'product_name': product_name,
                            'product_url': product_url,
                            'author': author,
                            'date': review_date,
                            'rating': rating,
                            'text': full_text
                        }

                        product_reviews.append(review_data)
                        self.all_reviews.append(review_data)

                except Exception:
                    continue

        except Exception:
            pass

        return product_reviews

    def save_reviews_to_txt(self, filename="flanco_reviews.txt"):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== REVIEW-URI PRODUSE FLANCO ===\n")
            f.write(f"Data extragerii: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
            f.write(f"Total review-uri: {len(self.all_reviews)}\n\n")

            for i, review in enumerate(self.all_reviews, 1):
                f.write(f"--- REVIEW #{i} ---\n")
                f.write(f"Produs: {review['product_name']}\n")
                f.write(f"URL: {review['product_url']}\n")
                f.write(f"Autor: {review['author']}\n")
                f.write(f"Data: {review['date']}\n")
                f.write(f"Rating: {review['rating']}\n")
                if 'likes' in review:
                    f.write(f"Like-uri: {review['likes']}\n")
                f.write(f"Text: {review['text']}\n\n")

        print(f"Review-urile au fost salvate in {filename}")

    def save_reviews_to_csv(self, filename="flanco_reviews.csv"):
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['product_name', 'product_url', 'author', 'date', 'rating', 'text']
            if self.all_reviews and 'likes' in self.all_reviews[0]:
                fieldnames.append('likes')

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for review in self.all_reviews:
                writer.writerow(review)

        print(f"Review-urile au fost salvate in {filename}")

    def save_products_to_csv(self, filename="flanco_products.csv"):
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['name', 'url', 'price', 'review_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for product in self.all_products:
                writer.writerow(product)

        print(f"Informatiile despre produse au fost salvate in {filename}")

    def run_extraction(self, laptop_pages=2, phone_pages=2, max_products=None):
        print("Colectez laptopuri...")
        laptop_urls = self.get_product_urls(
            'https://www.flanco.ro/laptop-it-tablete/laptop/dir/desc/order/reviews_count.html',
            max_pages=laptop_pages
        )

        print("Colectez telefoane...")
        phone_urls = self.get_product_urls(
            'https://www.flanco.ro/telefoane-tablete/smartphone/dir/desc/order/reviews_count.html',
            max_pages=phone_pages
        )

        all_urls = list(set(laptop_urls + phone_urls))

        if max_products and max_products < len(all_urls):
            all_urls = all_urls[:max_products]

        print(f"Incep extragerea review-urilor pentru {len(all_urls)} produse...")

        for url in tqdm(all_urls, desc="Procesare produse"):
            print(f"URL: {url}")
            reviews = self.extract_reviews_with_selenium(url)
            time.sleep(2)

        print(f"Extragere completa! S-au colectat {len(self.all_reviews)} review-uri in total.")

        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()

        return self.all_reviews, self.all_products


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extrage review-uri de pe Flanco.ro')

    parser.add_argument('--laptop-pages', type=int, default=2,
                        help='Numarul de pagini de laptopuri de analizat (default: 2)')

    parser.add_argument('--phone-pages', type=int, default=2,
                        help='Numarul de pagini de telefoane de analizat (default: 2)')

    parser.add_argument('--max-products', type=int, default=20,
                        help='Numarul maxim de produse de analizat (default: 20)')

    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directorul unde se vor salva rezultatele (default: directorul curent)')

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = FlancoReviewExtractor()

    try:
        reviews, products = extractor.run_extraction(
            laptop_pages=args.laptop_pages,
            phone_pages=args.phone_pages,
            max_products=args.max_products
        )

        extractor.save_reviews_to_txt(os.path.join(args.output_dir, 'flanco_reviews.txt'))
        extractor.save_reviews_to_csv(os.path.join(args.output_dir, 'flanco_reviews.csv'))
        extractor.save_products_to_csv(os.path.join(args.output_dir, 'flanco_products.csv'))

        print("\nStatistici extractie:")
        print(f"Total produse: {len(products)}")
        print(f"Total review-uri: {len(reviews)}")

    except Exception as e:
        print(f"Eroare in timpul extractiei: {str(e)}")
    finally:
        if hasattr(extractor, 'driver') and extractor.driver:
            extractor.driver.quit()


if __name__ == "__main__":
    main()