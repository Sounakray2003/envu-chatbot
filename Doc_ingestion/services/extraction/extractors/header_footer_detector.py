"""
Header/Footer Detector - PyMuPDF-based detection
Analyzes PDF pages to identify repeating header and footer zones
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# Try to import PyMuPDF
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available - header/footer detection disabled")


class HeaderFooterZone:
    """Container for header/footer exclusion zones"""
    
    def __init__(self):
        self.header_y_max: float = 0  # Pixels from top to exclude
        self.footer_y_min: float = 0  # Pixels from bottom to exclude
        self.page_height: float = 0
        self.confidence: float = 0.0
        self.header_patterns: List[str] = []
        self.footer_patterns: List[str] = []
    
    def __repr__(self) -> str:
        return (
            f"HeaderFooterZone(header={self.header_y_max:.1f}px, "
            f"footer={self.footer_y_min:.1f}px, "
            f"confidence={self.confidence:.2f})"
        )


class HeaderFooterDetector:
    """
    Detect header and footer zones in PDF using PyMuPDF
    
    Strategy:
    1. Sample first 3-5 and last 3-5 pages
    2. Extract text with coordinates
    3. Find repeating patterns in top/bottom margins
    4. Calculate exclusion zones
    """
    
    def __init__(
        self,
        header_zone_percent: float = 0.12,  # Top 12% of page
        footer_zone_percent: float = 0.12,  # Bottom 12% of page
        min_occurrence_rate: float = 0.5,   # Must appear on 50%+ of pages
        margin_px: int = 10                  # Extra margin around detected zones
    ):
        self.header_zone_percent = header_zone_percent
        self.footer_zone_percent = footer_zone_percent
        self.min_occurrence_rate = min_occurrence_rate
        self.margin_px = margin_px
    
    def detect(self, pdf_path: str, sample_pages: int = 5) -> Optional[HeaderFooterZone]:
        """
        Detect header/footer zones in PDF
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample from start and end
            
        Returns:
            HeaderFooterZone object or None if detection fails
        """
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not installed - skipping header/footer detection")
            return None
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                doc.close()
                return None
            
            # Determine sample indices
            sample_first = min(sample_pages, total_pages)
            sample_last = min(sample_pages, max(0, total_pages - sample_first))
            
            sample_indices = list(range(sample_first))
            if total_pages > sample_first:
                sample_indices.extend(
                    range(max(sample_first, total_pages - sample_last), total_pages)
                )
            
            logger.debug(f"Sampling {len(sample_indices)} pages from PDF with {total_pages} total pages")
            
            # Extract text with positions
            page_texts = []
            page_height = 0
            
            for page_num in sample_indices:
                page = doc[page_num]
                page_height = page.rect.height
                
                # Get text blocks with coordinates
                blocks = page.get_text("dict")["blocks"]
                
                page_texts.append({
                    'page_num': page_num,
                    'height': page_height,
                    'width': page.rect.width,
                    'blocks': blocks
                })
            
            doc.close()
            
            if not page_texts or page_height == 0:
                return None
            
            # Analyze top and bottom margins
            header_result = self._find_repeating_zone(
                page_texts, 'header', page_height
            )
            footer_result = self._find_repeating_zone(
                page_texts, 'footer', page_height
            )
            
            # Build zone object
            if header_result['y_boundary'] > 0 or footer_result['y_boundary'] > 0:
                zone = HeaderFooterZone()
                zone.header_y_max = header_result['y_boundary']
                zone.footer_y_min = footer_result['y_boundary']
                zone.page_height = page_height
                zone.confidence = (header_result['confidence'] + footer_result['confidence']) / 2
                zone.header_patterns = header_result.get('patterns', [])
                zone.footer_patterns = footer_result.get('patterns', [])
                
                logger.info(
                    f"Detected header/footer zones: "
                    f"header={zone.header_y_max:.1f}px, "
                    f"footer={zone.footer_y_min:.1f}px, "
                    f"confidence={zone.confidence:.2f}"
                )
                
                return zone
            
            logger.debug("No significant header/footer zones detected")
            return None
            
        except Exception as e:
            logger.warning(f"Header/footer detection failed: {e}")
            return None
    
    def _find_repeating_zone(
        self,
        page_texts: List[Dict],
        zone_type: str,
        page_height: float
    ) -> Dict:
        """
        Find repeating text patterns in header or footer zone
        
        Args:
            page_texts: List of page text data
            zone_type: 'header' or 'footer'
            page_height: Page height in points
            
        Returns:
            Dictionary with y_boundary, confidence, and patterns
        """
        # Define search zone
        if zone_type == 'header':
            zone_start_pct = 0.0
            zone_end_pct = self.header_zone_percent
        else:  # footer
            zone_start_pct = 1.0 - self.footer_zone_percent
            zone_end_pct = 1.0
        
        zone_start_y = page_height * zone_start_pct
        zone_end_y = page_height * zone_end_pct
        
        # Collect text from each page's zone
        zone_texts = {}
        
        for page_data in page_texts:
            page_num = page_data['page_num']
            texts_in_zone = []
            
            for block in page_data['blocks']:
                if block.get('type') != 0:  # Not text
                    continue
                
                # Get block position
                bbox = block.get('bbox', [0, 0, 0, 0])
                y_pos = bbox[1]  # Top y-coordinate
                
                # Check if in zone
                if zone_start_y <= y_pos <= zone_end_y:
                    # Extract text from lines
                    for line in block.get('lines', []):
                        for span in line.get('spans', []):
                            text = span.get('text', '').strip()
                            font_size = span.get('size', 12)
                            
                            if text and len(text) > 3:  # Ignore very short text
                                texts_in_zone.append({
                                    'text': text,
                                    'y': y_pos,
                                    'size': font_size
                                })
            
            if texts_in_zone:
                zone_texts[page_num] = texts_in_zone
        
        if not zone_texts:
            return {
                'y_boundary': 0,
                'confidence': 0.0,
                'patterns': []
            }
        
        # Find repeating patterns
        text_counts = {}
        y_positions = {}
        
        for page_num, texts in zone_texts.items():
            for item in texts:
                text = item['text']
                
                # Normalize text for comparison
                normalized = self._normalize_text(text)
                
                if not normalized:
                    continue
                
                if normalized not in text_counts:
                    text_counts[normalized] = 0
                    y_positions[normalized] = []
                
                text_counts[normalized] += 1
                y_positions[normalized].append(item['y'])
        
        # Find most common repeating text
        min_occurrences = max(2, len(page_texts) * self.min_occurrence_rate)
        repeating_texts = {
            text: count for text, count in text_counts.items()
            if count >= min_occurrences
        }
        
        if not repeating_texts:
            return {
                'y_boundary': 0,
                'confidence': 0.0,
                'patterns': []
            }
        
        # Calculate boundary
        all_y_positions = []
        for text in repeating_texts.keys():
            all_y_positions.extend(y_positions[text])
        
        if zone_type == 'header':
            # Header: exclude everything above the maximum y
            boundary = max(all_y_positions) + self.margin_px
        else:
            # Footer: exclude everything below the minimum y
            boundary = min(all_y_positions) - self.margin_px
        
        # Calculate confidence
        total_occurrences = sum(repeating_texts.values())
        max_possible = len(page_texts) * len(repeating_texts)
        confidence = min(total_occurrences / max_possible if max_possible > 0 else 0, 1.0)
        
        logger.debug(
            f"Found {len(repeating_texts)} repeating {zone_type} patterns "
            f"(boundary: y={boundary:.1f}, confidence: {confidence:.2f})"
        )
        
        return {
            'y_boundary': boundary,
            'confidence': confidence,
            'patterns': list(repeating_texts.keys())[:5]  # Top 5 patterns
        }
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for pattern matching
        
        Removes: page numbers, dates, extra whitespace
        """
        # Remove page numbers (standalone digits)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove dates
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', text)
        text = re.sub(r'\d{4}', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        text = text.lower().strip()
        
        # Remove common noise
        text = text.replace('page', '').replace('of', '').strip()
        
        return text
    
    def should_filter_text(
        self,
        text: str,
        y_position: float,
        zones: Optional[HeaderFooterZone]
    ) -> bool:
        """
        Check if text should be filtered based on position
        
        Args:
            text: Text content
            y_position: Y-coordinate of text
            zones: Detected header/footer zones
            
        Returns:
            True if text should be filtered
        """
        if not zones:
            return False
        
        # Check if in header zone
        if y_position <= zones.header_y_max:
            return True
        
        # Check if in footer zone
        if y_position >= zones.footer_y_min:
            return True
        
        return False
    
    def get_image_position(
        self,
        pdf_path: str,
        page_number: int,
        image_bytes: bytes
    ) -> Optional[Tuple[float, float]]:
        """
        Get y-coordinates of an image on a page
        
        Args:
            pdf_path: Path to PDF
            page_number: Page number (1-indexed)
            image_bytes: Image bytes to locate
            
        Returns:
            Tuple of (y_top, y_bottom) or None
        """
        if not PYMUPDF_AVAILABLE:
            return None
        
        try:
            doc = fitz.open(pdf_path)
            
            if page_number < 1 or page_number > len(doc):
                doc.close()
                return None
            
            page = doc[page_number - 1]
            
            # Get all images on page
            image_list = page.get_images()
            
            # Try to find matching image and get position
            for img_info in image_list:
                xref = img_info[0]
                
                # Get image rectangles
                rect_list = page.get_image_rects(xref)
                
                if rect_list:
                    rect = rect_list[0]  # Use first occurrence
                    doc.close()
                    return (rect.y0, rect.y1)
            
            doc.close()
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get image position: {e}")
            return None
    
    def is_image_in_zone(
        self,
        pdf_path: str,
        page_number: int,
        image_bytes: bytes,
        zones: Optional[HeaderFooterZone]
    ) -> bool:
        """
        Check if image is in header/footer zone
        
        Args:
            pdf_path: Path to PDF
            page_number: Page number (1-indexed)
            image_bytes: Image bytes
            zones: Detected zones
            
        Returns:
            True if image is in header/footer zone
        """
        if not zones:
            return False
        
        position = self.get_image_position(pdf_path, page_number, image_bytes)
        
        if not position:
            return False
        
        y_top, y_bottom = position
        
        # Check if in header zone
        if y_bottom <= zones.header_y_max:
            return True
        
        # Check if in footer zone
        if y_top >= zones.footer_y_min:
            return True
        
        return False


# Singleton instance
_detector = None


def get_header_footer_detector() -> HeaderFooterDetector:
    """Get singleton instance of HeaderFooterDetector"""
    global _detector
    if _detector is None:
        _detector = HeaderFooterDetector()
    return _detector