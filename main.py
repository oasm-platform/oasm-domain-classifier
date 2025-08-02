from crawl_web import CrawlWeb
import json
from domain_classifier import DomainClassifier


def main():
    url = "https://dichvucong.gov.vn/p/home/dvc-trang-chu.html"

    classifier = DomainClassifier()
    crawler = CrawlWeb()
    text = crawler.crawl(url)

    print(f"Crawled text: {text}")

    result = classifier.predict(text)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Predicted domain: {result['predicted_domain']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nAll probabilities:")
    print(json.dumps(result['all_probabilities'], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
