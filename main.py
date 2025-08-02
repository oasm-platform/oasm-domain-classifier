from crawl_web import CrawlWeb

def main(url: str):
    crawler = CrawlWeb()
    result = crawler.crawl(url)
    if result:
        print("Successfully crawled content:")
        print(result)
    else:
        print("Failed to crawl the page.")
    return result

if __name__ == "__main__":
    url = "https://www.24h.com.vn"
    main(url)