"""
Knowledge Extractor Module
Extracts subject-predicate-object triples from text using either LLM or spaCy
"""

import json
import re
from typing import List, Dict
import spacy
from groq import Groq


class KnowledgeExtractor:
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the knowledge extractor
        
        Args:
            groq_api_key: API key for Groq (optional, needed for LLM mode)
        """
        self.groq_api_key = groq_api_key
        self.nlp = None
        
    def _load_spacy_model(self):
        """Load spaCy model lazily"""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise Exception(
                    "spaCy model not found. Please run: "
                    "python -m spacy download en_core_web_sm"
                )
    
    def extract_with_llm(self, text: str) -> List[Dict[str, str]]:
        """
        Extract knowledge triples using Groq LLM
        
        Args:
            text: Input text to extract knowledge from
            
        Returns:
            List of dictionaries with 'subject', 'predicate', 'object' keys
        """
        if not self.groq_api_key:
            raise ValueError("Groq API key is required for LLM extraction")
        
        client = Groq(api_key=self.groq_api_key)
        
        prompt = f"""Extract knowledge triples from the following text. 
Return your response as a JSON array of objects, where each object has exactly three fields: 
"subject", "predicate", and "object".

Example format:
[
  {{"subject": "Paris", "predicate": "is capital of", "object": "France"}},
  {{"subject": "Einstein", "predicate": "developed", "object": "theory of relativity"}}
]

Text to analyze:
{text}

Return only the JSON array, no additional text or explanation."""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledge extraction expert. Extract subject-predicate-object triples and return them as valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content
            
            # Clean the response - remove markdown code blocks if present
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
            result = result.strip()
            
            # Parse JSON
            triples = json.loads(result)
            
            # Validate structure
            if not isinstance(triples, list):
                triples = [triples]
                
            # Ensure all triples have required keys
            valid_triples = []
            for triple in triples:
                if all(key in triple for key in ['subject', 'predicate', 'object']):
                    valid_triples.append({
                        'subject': str(triple['subject']).strip(),
                        'predicate': str(triple['predicate']).strip(),
                        'object': str(triple['object']).strip()
                    })
            
            return valid_triples
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {result}")
            return []
        except Exception as e:
            print(f"Error during LLM extraction: {e}")
            return []
    
    def extract_with_spacy(self, text: str) -> List[Dict[str, str]]:
        """
        Extract knowledge triples using spaCy NLP
        
        Args:
            text: Input text to extract knowledge from
            
        Returns:
            List of dictionaries with 'subject', 'predicate', 'object' keys
        """
        self._load_spacy_model()
        
        doc = self.nlp(text)
        triples = []
        
        for sent in doc.sents:
            subject = None
            predicate = None
            obj = None
            
            # Find the root verb (predicate)
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    predicate = token.lemma_
                    
                    # Find subject
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subject = self._get_full_phrase(child)
                        
                        # Find object
                        elif child.dep_ in ("dobj", "attr", "oprd"):
                            obj = self._get_full_phrase(child)
                        
                        # Find prepositional objects
                        elif child.dep_ == "prep":
                            for prep_child in child.children:
                                if prep_child.dep_ == "pobj":
                                    if not obj:  # Only use if we don't have a direct object
                                        obj = self._get_full_phrase(prep_child)
                                        predicate = f"{predicate} {child.text}"
            
            # Add triple if we have all components
            if subject and predicate and obj:
                triples.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj
                })
        
        return triples
    
    def _get_full_phrase(self, token) -> str:
        """
        Get the full phrase including modifiers for a token
        
        Args:
            token: spaCy token
            
        Returns:
            Full phrase as string
        """
        # Get all children tokens
        phrase_tokens = [token]
        
        for child in token.children:
            if child.dep_ in ("det", "amod", "compound", "poss"):
                phrase_tokens.append(child)
        
        # Sort by position in sentence
        phrase_tokens.sort(key=lambda t: t.i)
        
        # Join tokens
        phrase = " ".join([t.text for t in phrase_tokens])
        
        return phrase
    
    def extract(self, text: str, mode: str = "llm") -> List[Dict[str, str]]:
        """
        Extract knowledge triples using specified mode
        
        Args:
            text: Input text to extract knowledge from
            mode: Extraction mode - 'llm' or 'spacy'
            
        Returns:
            List of dictionaries with 'subject', 'predicate', 'object' keys
        """
        if not text or not text.strip():
            return []
        
        if mode.lower() == "llm":
            return self.extract_with_llm(text)
        elif mode.lower() == "spacy":
            return self.extract_with_spacy(text)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'llm' or 'spacy'")


if __name__ == "__main__":
    # Test the extractor
    test_text = """
    Albert Einstein was a German physicist who developed the theory of relativity.
    He was born in Ulm, Germany in 1879.
    Einstein won the Nobel Prize in Physics in 1921.
    """
    
    # Test spaCy extraction
    extractor = KnowledgeExtractor()
    print("Testing spaCy extraction:")
    triples = extractor.extract(test_text, mode="spacy")
    for triple in triples:
        print(f"  {triple}")