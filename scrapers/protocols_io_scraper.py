import re
import time
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.models.scraped_experiment import (
    ScrapedExperiment,
    ExperimentConditions,
    ExperimentOutcome,
    ExperimentEvidence,
)
class ProtocolsIOScraper:
    SEARCH_API = "https://www.protocols.io/api/v3/protocols"
    PROTOCOL_API = "https://www.protocols.io/api/v4/protocols/{}"
    def __init__(self, api_token: str = "267aed2f9747b4f5d21c3515a8701172d74b12afe342a9a75c23aaa3f7fda8d0ab30bc9986feaed12851c4673bc34711d69b8a5ccec46474cdc7f3d16b9d55e4"):
        self.api_token = api_token
    def _extract_conditions_from_text(self, text: str) -> Dict[str, Any]:
        conditions = {}
        temp_patterns = [
            r'(\-?\d+(?:\.\d+)?)\s*°?\s*C',
            r'(\-?\d+(?:\.\d+)?)\s*degrees?\s*Celsius',
            r'at\s+(\-?\d+(?:\.\d+)?)\s*°?\s*C',
            r'(\-?\d+(?:\.\d+)?)\s*°?\s*C\s+for',
        ]
        temperatures = []
        for pattern in temp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temperatures.extend([float(t) for t in matches])
        if temperatures:
            conditions['temperature'] = temperatures[0]
        ph_patterns = [
            r'pH\s*(\d+(?:\.\d+)?)',
            r'pH\s*=\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in ph_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                conditions['ph'] = float(matches[0])
                break
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:hours?|h)\b',
            r'(\d+(?:\.\d+)?)\s*(?:minutes?|min)\b',
            r'(\d+(?:\.\d+)?)\s*(?:seconds?|sec|s)\b',
            r'(\d+(?:\.\d+)?)\s*(?:days?|d)\b',
        ]
        durations = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            durations.extend(matches)
        if durations:
            conditions['duration'] = durations[0]
        conc_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:NaCl|salt|saline)',
            r'(\d+(?:\.\d+)?)\s*(?:mg/mL|μg/mL|ng/mL|mg/L|μg/L)',
        ]
        for pattern in conc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                conditions['concentration'] = matches[0]
                break
        volume_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mL|μL|uL|L|liters?)',
        ]
        volumes = []
        for pattern in volume_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            volumes.extend(matches)
        if volumes:
            conditions['volume'] = volumes[0]
        organism_patterns = [
            r'(?:Bacillus|Escherichia|E\.?\s*coli|Staphylococcus|Salmonella|Pseudomonas|Saccharomyces)',
            r'(?:bacteria|bacterial|microorganism|microbe)',
        ]
        for pattern in organism_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if 'organism' not in conditions:
                    conditions['organism'] = re.search(pattern, text, re.IGNORECASE).group()
                break
        return conditions
    def _parse_step_content(self, step_data: str) -> str:
        try:
            if isinstance(step_data, str):
                parsed = json.loads(step_data)
                if isinstance(parsed, dict) and 'blocks' in parsed:
                    texts = []
                    for block in parsed['blocks']:
                        if 'text' in block and block['text'].strip():
                            texts.append(block['text'].strip())
                    return ' '.join(texts)
            return step_data if isinstance(step_data, str) else str(step_data)
        except:
            return str(step_data)
    def _search_protocols(self, query: str, limit: int) -> List[dict]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "biomemory/0.1"
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        else:
            print("No API token available")
        params = {
            "filter": "public",
            "key": query,
            "page_size": min(limit, 100),
            "page_id": 1
        }
        try:
            r = requests.get(
                self.SEARCH_API,
                headers=headers,
                params=params,
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            protocols = data.get("items", [])
            return protocols[:limit]
        except Exception as e:
            print(f"Error searching protocols: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return []
    def _fetch_protocol(self, protocol_uri: str) -> dict:
        headers = {
            "Accept": "application/json",
            "User-Agent": "biomemory/0.1"
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        try:
            r = requests.get(
                self.PROTOCOL_API.format(protocol_uri),
                headers=headers,
                timeout=20
            )
            r.raise_for_status()
            data = r.json()
            return data.get("payload", {})
        except Exception as e:
            print(f"Error fetching protocol {protocol_uri}: {e}")
            return {}
    def scrape(self, query: str, limit: int = 5) -> List[ScrapedExperiment]:
        protocols = self._search_protocols(query, limit)
        experiments = []
        print(f"Found {len(protocols)} protocols")
        for protocol in protocols:
            try:
                protocol_id = protocol.get("id") or protocol.get("uri")
                if not protocol_id:
                    continue
                item = self._fetch_protocol(protocol_id)
                if not item:
                    continue
                extracted_conditions = {}
                full_description = item.get("description", "")
                if "steps" in item and item["steps"]:
                    all_step_texts = []
                    for step in item["steps"]:
                        if isinstance(step, dict):
                            step_text = self._parse_step_content(step.get("step", ""))
                            if step_text:
                                all_step_texts.append(step_text)
                                step_conditions = self._extract_conditions_from_text(step_text)
                                extracted_conditions.update(step_conditions)
                    full_description += " " + " ".join(all_step_texts)
                if item.get("guidelines"):
                    guidelines_text = self._parse_step_content(item["guidelines"])
                    guidelines_conditions = self._extract_conditions_from_text(guidelines_text)
                    extracted_conditions.update(guidelines_conditions)
                    full_description += " " + guidelines_text
                if item.get("before_start"):
                    before_text = self._parse_step_content(item["before_start"])
                    before_conditions = self._extract_conditions_from_text(before_text)
                    extracted_conditions.update(before_conditions)
                    full_description += " " + before_text
                materials_info = ""
                if item.get("materials_text"):
                    materials_info = item["materials_text"]
                    full_description += " " + materials_info
                experiment = ScrapedExperiment(
                    experiment_id=f"protocols_io_{protocol_id}",
                    description=full_description[:2000],
                    sequence="",
                    conditions=ExperimentConditions(
                        ph=extracted_conditions.get("ph"),
                        temperature=extracted_conditions.get("temperature"),
                        organism=extracted_conditions.get("organism") or item.get("organism_name"),
                        assay=item.get("protocol_type"),
                        additional={
                            "category": item.get("category"),
                            "keywords": item.get("keywords", []),
                            "duration": extracted_conditions.get("duration"),
                            "concentration": extracted_conditions.get("concentration"),
                            "volume": extracted_conditions.get("volume"),
                            "materials": materials_info,
                            "steps_count": len(item.get("steps", [])),
                        }
                    ),
                    outcome=ExperimentOutcome(
                        status="unknown",
                        notes=f"Scraped from protocols.io API - {len(item.get('steps', []))} steps",
                        metrics={}
                    ),
                    evidence=ExperimentEvidence(
                        paper_title=item.get("title"),
                        doi=item.get("doi"),
                        arxiv_link=None,
                        protocol_url=f"https://www.protocols.io/view/{protocol_id}",
                        authors=[
                            a.get("name")
                            for a in item.get("authors", [])
                            if isinstance(a, dict) and "name" in a
                        ],
                        publication_date=str(item.get("created_on")) if item.get("created_on") else None
                    ),
                    source="protocols.io",
                    scraped_at=datetime.utcnow()
                )
                experiments.append(experiment)
                time.sleep(1)
            except Exception as e:
                print(f"Error with protocol {protocol_id}: {e}")
        return experiments

    def _scrape_protocol_page_bs4(self, protocol_url: str) -> Optional[Dict[str, Any]]:
        """Scrape a protocols.io protocol page using BeautifulSoup as fallback/enrichment."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; biomemory/0.1)",
                "Accept": "text/html,application/xhtml+xml",
            }
            r = requests.get(protocol_url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            title = ""
            title_tag = soup.find("h1") or soup.find("meta", {"property": "og:title"})
            if title_tag:
                title = title_tag.get_text(strip=True) if title_tag.name == "h1" else title_tag.get("content", "")

            description = ""
            desc_meta = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
            if desc_meta:
                description = desc_meta.get("content", "")

            # Extract steps from protocol content
            steps_text = []
            step_elements = soup.find_all(["div", "section"], class_=lambda c: c and "step" in str(c).lower())
            if not step_elements:
                step_elements = soup.find_all("div", class_=lambda c: c and "protocol" in str(c).lower())
            for el in step_elements:
                txt = el.get_text(separator=" ", strip=True)
                if len(txt) > 20:
                    steps_text.append(txt)

            # If no structured steps found, extract from main content
            if not steps_text:
                main = soup.find("main") or soup.find("article") or soup.find("div", {"id": "protocol"})
                if main:
                    for p in main.find_all(["p", "li"]):
                        txt = p.get_text(strip=True)
                        if len(txt) > 20:
                            steps_text.append(txt)

            full_text = f"{title} {description} {' '.join(steps_text)}"
            conditions = self._extract_conditions_from_text(full_text)

            # Extract authors
            authors = []
            author_els = soup.find_all("a", class_=lambda c: c and "author" in str(c).lower())
            if not author_els:
                author_els = soup.find_all("meta", {"name": "author"})
            for a in author_els:
                name = a.get_text(strip=True) if a.name == "a" else a.get("content", "")
                if name:
                    authors.append(name)

            # Extract DOI
            doi = None
            doi_tag = soup.find("a", href=lambda h: h and "doi.org" in h)
            if doi_tag:
                doi = doi_tag.get("href", "").replace("https://doi.org/", "")

            return {
                "title": title,
                "description": full_text[:2000],
                "steps_text": steps_text,
                "conditions": conditions,
                "authors": authors,
                "doi": doi,
                "url": protocol_url,
                "steps_count": len(steps_text),
            }
        except Exception as e:
            print(f"BS4 scrape failed for {protocol_url}: {e}")
            return None

    def scrape_bs4(self, query: str, limit: int = 5) -> List[ScrapedExperiment]:
        """Scrape protocols using BeautifulSoup by first searching via API then parsing HTML pages."""
        protocols = self._search_protocols(query, limit)
        experiments = []
        print(f"[BS4] Found {len(protocols)} protocols, scraping HTML pages...")

        for protocol in protocols:
            try:
                protocol_id = protocol.get("id") or protocol.get("uri")
                if not protocol_id:
                    continue

                protocol_url = f"https://www.protocols.io/view/{protocol_id}"
                data = self._scrape_protocol_page_bs4(protocol_url)
                if not data or len(data.get("description", "")) < 30:
                    continue

                extracted = data["conditions"]

                experiment = ScrapedExperiment(
                    experiment_id=f"protocols_io_bs4_{protocol_id}",
                    description=data["description"],
                    sequence="",
                    conditions=ExperimentConditions(
                        ph=extracted.get("ph"),
                        temperature=extracted.get("temperature"),
                        organism=extracted.get("organism"),
                        assay=protocol.get("protocol_type"),
                        additional={
                            "duration": extracted.get("duration"),
                            "concentration": extracted.get("concentration"),
                            "volume": extracted.get("volume"),
                            "steps_count": data["steps_count"],
                            "scrape_method": "beautifulsoup",
                        }
                    ),
                    outcome=ExperimentOutcome(
                        status="unknown",
                        notes=f"Scraped via BeautifulSoup - {data['steps_count']} steps extracted",
                        metrics={}
                    ),
                    evidence=ExperimentEvidence(
                        paper_title=data.get("title"),
                        doi=data.get("doi"),
                        arxiv_link=None,
                        protocol_url=protocol_url,
                        authors=data.get("authors", []),
                        publication_date=None
                    ),
                    source="protocols.io",
                    scraped_at=datetime.utcnow()
                )
                experiments.append(experiment)
                time.sleep(1.5)
            except Exception as e:
                print(f"[BS4] Error with protocol {protocol_id}: {e}")

        print(f"[BS4] Successfully scraped {len(experiments)} experiments")
        return experiments