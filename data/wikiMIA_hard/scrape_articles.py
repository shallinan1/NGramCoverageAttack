# Wikipedia article scraper for membership inference attack (MIA) research
# This script compares Wikipedia article summaries from 2016 vs current versions
# to identify articles that have changed significantly over time

import requests
import time
from wikitextparser import remove_markup, parse
import re
from tqdm import tqdm
import Levenshtein
from datetime import datetime
from unidecode import unidecode
import json
import argparse
import os
import multiprocessing as mp
from multiprocessing import Queue
import threading

# Create a persistent session for API requests to Wikipedia
session = requests.Session()
BASE_URL = "https://en.wikipedia.org/w/api.php"

def is_stub(title):
    """
    Check if a Wikipedia article is a stub (short, incomplete article)
    by examining its categories for any containing 'stubs'
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "categories",
        "titles": title,
        "cllimit": "max"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    if "categories" not in page:
        return False
    return any("stubs" in cat["title"].lower() for cat in page["categories"])

def remove_urls(text):
    """Remove URLs from text to clean up article summaries"""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def get_random_article():
    """
    Get a random Wikipedia article title, filtering out unwanted types:
    - List articles (e.g., "List of countries")
    - Disambiguation pages
    - Stub articles (short, incomplete articles)
    
    Continues searching until it finds a suitable main namespace article
    """
    while True:
        # Get a random article from main namespace (0)
        res = session.get(BASE_URL, params={
            "format": "json",
            "action": "query",
            "list": "random",
            "rnnamespace": 0,  # Main namespace only
            "rnlimit": 1,
        })
        title = res.json()["query"]["random"][0]["title"]
        
        # Skip unwanted article types
        if (
            title.startswith("List of") or
            "disambiguation" in title.lower() or
            is_stub(title)
        ):
            debug_print(f"[SKIPPED] {title} — list/disambig/stub")
            continue
        
        # Double-check for disambiguation pages via pageprops
        page_info = session.get(BASE_URL, params={
            "format": "json",
            "action": "query",
            "prop": "pageprops",
            "titles": title,
        }).json()
        page = next(iter(page_info["query"]["pages"].values()))
        if "pageprops" in page and "disambiguation" in page["pageprops"]:
            continue

        return title

def get_revision_id_as_of(title, date="2016-12-31T23:59:59Z"):
    """
    Get the revision ID of an article as it existed on a specific date
    Default date is end of 2016 for the membership inference study
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",  # Get revisions going backwards in time
        "rvstart": date,   # Start from this date and go backwards
        "rvprop": "ids",   # Only need the revision ID
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    if "revisions" not in page:
        return None
    return page["revisions"][0]["revid"]

def get_revision_wikitext(revid):
    """
    Get the raw wikitext content for a specific revision ID
    Returns the markup text that Wikipedia uses internally
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "revids": revid,
        "rvprop": "content",
        "rvslots": "main"  # Get content from main slot
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    try:
        output = page["revisions"][0]["slots"]["main"]["*"]
        return output
    except:
        return None

def get_latest_revision(title):
    """
    Get the current version of an article's wikitext and its timestamp
    Returns both the content and when it was last modified
    """
    res = session.get(BASE_URL, params={
        "format": "json",
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,  # Only get the most recent revision
        "rvprop": "content|timestamp",  # Get both content and edit timestamp
        "rvslots": "main"
    })
    page = next(iter(res.json()["query"]["pages"].values()))
    revision = page["revisions"][0]
    wikitext = revision["slots"]["main"]["*"]
    timestamp = revision["timestamp"]
    return wikitext, timestamp

def extract_plain_summary(wikitext):
    """
    Convert Wikipedia wikitext markup to plain text summary
    Extracts only the first section (intro/summary) and cleans it up
    
    Filters out articles with:
    - Empty or very short summaries (< 5 words)
    - Summaries that start with "is a" (usually parsing errors)
    - Parsing artifacts like "accessed on" or "thumb" remnants
    """
    try:
        parsed = parse(wikitext)
    except:
        return ""

    parsed = parse(wikitext)

    # Extract first section (intro/summary) as plain text
    summary = parsed.sections[0].plain_text().strip()
    summary = re.sub(r' {2,}', ' ', summary)  # Collapse multiple spaces

    # Clean up common parsing artifacts from images/thumbnails
    if summary.startswith("thumb"):
        index = summary.find("\n")
        if index != -1:
            summary = summary[index+1:].strip()
    if "\n\n\nthumb" in summary:
        index = summary.find("\n\n\nthumb")
        if index != 1:
            summary =  summary[:index].strip()

    if summary == "":
        return ""
    
    # Remove URLs and empty parentheses
    summary = remove_urls(summary).strip()
    summary = re.sub(r'\(\s*[\.,;:\-!?]*\s*\)', '', summary)

    # Filter out low-quality summaries
    if summary.startswith("is a"):
        return ""  # Usually indicates parsing error
    
    if len(summary.split()) < 5:
        return ""  # Too short to be meaningful

    if "accessed on" in summary:
        return ""  # Parsing error artifact

    # Convert Unicode characters to ASCII equivalents
    return unidecode(summary)

# Global variables for the scraping process
DEBUG_MODE = False  # Global debug flag

# Target date for historical comparison (end of 2016)
raw_date = "2016-12-31T23:59:59Z"
parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled"""
    if DEBUG_MODE:
        print(*args, **kwargs)

def compare_summaries(title):
    """
    Compare Wikipedia article summaries from 2016 vs current version
    
    This is the core function that:
    1. Gets the 2016 version of the article
    2. Gets the current version 
    3. Extracts and compares the summaries
    4. Calculates similarity metrics
    5. Filters for articles with recent edits (2024+)
    
    Returns article data if changed significantly, False otherwise
    """
    # Get the revision ID from 2016
    revid = get_revision_id_as_of(title)
    
    if revid is None:
        return False  # Article didn't exist in 2016
    
    # Get the old wikitext content
    old_wikitext = get_revision_wikitext(revid)
    if old_wikitext is None:
        return False  # Couldn't retrieve old content
        
    # Get current version
    new_wikitext, timestamp = get_latest_revision(title)

    # Only include articles that have been edited recently (2024+)
    # This ensures we're looking at articles that are actively maintained
    latest_revision_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if latest_revision_date < datetime(2024, 1, 1, tzinfo=latest_revision_date.tzinfo):
        debug_print(f"[SKIPPED: TOO OLD] {title} — Last edit on {latest_revision_date.date()}")
        return False

    # Extract clean summaries from both versions
    old_summary = extract_plain_summary(old_wikitext)
    new_summary = extract_plain_summary(new_wikitext)

    # Skip if either summary is empty/invalid
    if old_summary == "" or new_summary == "":
        return False

    # Check if the summaries have changed
    if old_summary == new_summary:
        debug_print(f"[UNCHANGED] {title}")
        return False
    else:
        # Calculate edit distance between summaries
        diff_chars = Levenshtein.distance(old_summary, new_summary)

        debug_print(f"[CHANGED] {title} — Levenshtein Distance: {diff_chars} characters")

        # Return structured data about the changes
        return {
            "title": title,
            "old_summary": old_summary,
            "new_summary": new_summary,
            "char_difference": diff_chars,
            "percent_diff": (2*diff_chars)/(len(old_summary) + len(new_summary)),
            "first_retrieved_date": parsed_date.date().isoformat(),
            "last_edit_date": latest_revision_date.date().isoformat()
        }

def worker_process(worker_id, articles_per_worker, results_queue):
    """
    Worker process function that scrapes articles and puts results in a queue
    Each worker has its own requests session to avoid conflicts
    """
    # Create a new session for this worker process
    worker_session = requests.Session()
    
    # Override the global session for this worker
    global session
    session = worker_session
    
    count = 0  # Articles found by this worker
    
    # Create progress bar for this worker
    pbar = tqdm(range(articles_per_worker), desc=f"Worker {worker_id}", position=int(worker_id.split('_')[1]))
    
    for i in pbar:
        try:
            title = get_random_article()
            print(title)
            result = compare_summaries(title)
            
            if result:
                count += 1
                # Send result to main process via queue
                results_queue.put(result)
            
            # Update progress bar with hit rate
            processed = i + 1
            hit_rate = (count / processed * 100) if processed > 0 else 0
            pbar.set_postfix(hits=count, rate=f"{hit_rate:.1f}%")
            
            # Rate limiting - be nice to Wikipedia's servers
            time.sleep(0.1)
            
        except Exception as e:
            debug_print(f"Worker {worker_id} error: {e}")
            continue
    
    pbar.close()
    print(f"Worker {worker_id} complete: found {count} articles")

def file_writer_thread(results_queue, output_file, total_expected):
    """
    Background thread that writes results from queue to file
    This prevents blocking the worker processes
    """
    results_written = 0
    
    with open(output_file, "a") as f:
        while results_written < total_expected:
            try:
                # Get result from queue (with timeout to check for completion)
                result = results_queue.get(timeout=30)
                if result is None:  # Sentinel value to stop
                    break
                    
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure data is written immediately
                results_written += 1
                
                if results_written % 10 == 0:
                    debug_print(f"Written {results_written} results to {output_file}")
                    
            except:
                # Timeout - check if all workers are done
                break

def main():
    """Main function to handle command line arguments and run the scraper"""
    parser = argparse.ArgumentParser(description='Scrape Wikipedia articles for membership inference research')
    parser.add_argument('--max-tries', type=int, default=200000,
                       help='Maximum number of articles to examine (default: 200000)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--output-file', type=str, default="data/wikiMIA_hard/scraped/scraped.jsonl",
                       help='Output file path (default: data/wikiMIA_hard/scraped/scraped.jsonl)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (shows skipped/unchanged articles)')
    args = parser.parse_args()
    
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Wikipedia article scraping with {args.workers} parallel workers")
    print(f"Looking for articles changed between 2016 and present (with 2024+ edits)")
    print(f"Will examine {args.max_tries} total articles ({args.max_tries // args.workers} per worker)")
    print(f"Output file: {args.output_file}")
    
    # Calculate work distribution
    articles_per_worker = args.max_tries // args.workers
    
    # Create queue for results communication between workers and main process
    results_queue = Queue()
    
    # Start background thread to write results to file
    writer_thread = threading.Thread(
        target=file_writer_thread,
        args=(results_queue, args.output_file, args.max_tries)
    )
    writer_thread.daemon = True
    writer_thread.start()
    
    # Create and start worker processes
    processes = []
    for i in range(args.workers):
        p = mp.Process(
            target=worker_process,
            args=(f"worker_{i}", articles_per_worker, results_queue)
        )
        processes.append(p)
        p.start()
        print(f"Started worker {i}")
    
    for p in processes:
        p.join()
    
    # Signal file writer thread to stop and wait for it
    results_queue.put(None)  # Sentinel value
    writer_thread.join(timeout=10)
    
    print("All workers complete!")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()

"""
To run this script:

Basic usage (4 workers, respects rate limits):
python3 -m data.wikiMIA_hard.scrape_articles

Custom configuration:
python3 -m data.wikiMIA_hard.scrape_articles --workers 8 --max-tries 1000000

Note: Each worker waits 0.1 seconds between requests to respect Wikipedia's rate limits.
With 4 workers, this means ~40 requests per second across all workers.
Adjust --workers based on your needs, but be mindful of Wikipedia's servers.
"""
