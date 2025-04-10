from typing import Dict, List, TypedDict

class AestheticComponents(TypedDict):
    style_elements: List[str]
    color_palette: List[str]
    key_brands: List[str]
    silhouettes: List[str]

AESTHETIC_CATEGORIES: Dict[str, AestheticComponents] = {
    "Minimalist": {
        "style_elements": [
            "Clean lines",
            "Basic essentials",
            "Quality fabrics",
            "Subtle details",
            "Timeless pieces",
            "Monochromatic looks",
            "Minimal accessories"
        ],
        "color_palette": [
            "#000000",  # Black
            "#FFFFFF",  # White
            "#E5E5E5",  # Light Gray
            "#9A8A78",  # Beige
            "#1B365D"   # Navy
        ],
        "key_brands": [
            "COS",
            "Uniqlo",
            "Everlane",
            "Arket",
            "Theory",
            "& Other Stories"
        ],
        "silhouettes": [
            "Straight cut",
            "Boxy fit",
            "A-line",
            "Relaxed fit",
            "Architectural shapes"
        ]
    },
    "Dark Academia": {
        "style_elements": [
            "Tweed fabrics",
            "Plaid patterns",
            "Leather accessories",
            "Vintage books",
            "Knit sweaters",
            "Oxford shoes",
            "Gold-rimmed glasses"
        ],
        "color_palette": [
            "#4A3728",  # Dark Brown
            "#2B331F",  # Forest Green
            "#800020",  # Burgundy
            "#1B365D",  # Navy
            "#E8DCC4"   # Cream
        ],
        "key_brands": [
            "Ralph Lauren",
            "Brooks Brothers",
            "Burberry",
            "J.Crew",
            "Pendleton"
        ],
        "silhouettes": [
            "Blazers",
            "Pleated skirts",
            "Oxford shirts",
            "Sweater vests",
            "Tailored trousers"
        ]
    },
    "Grunge": {
        "style_elements": [
            "Distressed denim",
            "Flannel shirts",
            "Band t-shirts",
            "Combat boots",
            "Oversized sweaters",
            "Chain accessories",
            "Leather jackets"
        ],
        "color_palette": [
            "#000000",  # Black
            "#4A4A4A",  # Dark Gray
            "#8B0000",  # Dark Red
            "#4B0082",  # Indigo
            "#2F4F4F"   # Dark Slate
        ],
        "key_brands": [
            "Dr. Martens",
            "Converse",
            "Vans",
            "Levi's",
            "Dickies"
        ],
        "silhouettes": [
            "Oversized",
            "Layered",
            "Loose-fitting",
            "Deconstructed",
            "Raw-hemmed"
        ]
    },
    "Y2K": {
        "style_elements": [
            "Crop tops",
            "Platform shoes",
            "Metallic fabrics",
            "Butterfly motifs",
            "Low-rise jeans",
            "Chunky jewelry",
            "Mini bags"
        ],
        "color_palette": [
            "#FF1493",  # Deep Pink
            "#4B0082",  # Indigo
            "#FFD700",  # Gold
            "#00FFFF",  # Cyan
            "#FF69B4"   # Hot Pink
        ],
        "key_brands": [
            "Juicy Couture",
            "Von Dutch",
            "Fendi",
            "Baby Phat",
            "Bebe"
        ],
        "silhouettes": [
            "Bodycon",
            "Cropped",
            "Low-rise",
            "Mini length",
            "Tank style"
        ]
    },
    "Cottagecore": {
        "style_elements": [
            "Floral prints",
            "Lace details",
            "Natural fabrics",
            "Puff sleeves",
            "Ruffles",
            "Vintage-inspired",
            "Straw accessories"
        ],
        "color_palette": [
            "#F5E6E8",  # Light Pink
            "#D4E2D4",  # Sage Green
            "#FAF2E4",  # Cream
            "#C6A5A5",  # Dusty Rose
            "#E8DCC4"   # Beige
        ],
        "key_brands": [
            "Christy Dawn",
            "Hill House Home",
            "Faithfull the Brand",
            "Doen",
            "Reformation"
        ],
        "silhouettes": [
            "A-line dresses",
            "Puff sleeves",
            "Tiered skirts",
            "Empire waist",
            "Gathered details"
        ]
    },
    "Streetwear": {
        "style_elements": [
            "Graphic t-shirts",
            "Sneakers",
            "Hoodies",
            "Cargo pants",
            "Baseball caps",
            "Statement outerwear",
            "Logo prints"
        ],
        "color_palette": [
            "#000000",  # Black
            "#FF0000",  # Red
            "#FFFFFF",  # White
            "#808080",  # Gray
            "#FFD700"   # Gold
        ],
        "key_brands": [
            "Supreme",
            "Off-White",
            "BAPE",
            "Nike",
            "Stussy"
        ],
        "silhouettes": [
            "Oversized",
            "Loose fit",
            "Layered",
            "Boxy",
            "Utilitarian"
        ]
    },
    "Dark Luxury": {
        "style_elements": [
            "High-end materials",
            "Metallic accents",
            "Statement jewelry",
            "Leather pieces",
            "Sleek silhouettes",
            "Designer logos",
            "Fur details"
        ],
        "color_palette": [
            "#000000",  # Black
            "#1A1A1A",  # Charcoal
            "#4A4A4A",  # Dark Gray
            "#B8860B",  # Dark Gold
            "#800020"   # Burgundy
        ],
        "key_brands": [
            "Saint Laurent",
            "Balenciaga",
            "Tom Ford",
            "Givenchy",
            "Alexander McQueen"
        ],
        "silhouettes": [
            "Fitted",
            "Structured",
            "Sharp shoulders",
            "Slim cut",
            "Dramatic lines"
        ]
    },
    "Athleisure": {
        "style_elements": [
            "Performance fabrics",
            "Sports bras",
            "Leggings",
            "Track jackets",
            "Mesh details",
            "Technical sneakers",
            "Moisture-wicking materials"
        ],
        "color_palette": [
            "#000000",  # Black
            "#FFFFFF",  # White
            "#FF69B4",  # Pink
            "#4169E1",  # Royal Blue
            "#808080"   # Gray
        ],
        "key_brands": [
            "Lululemon",
            "Nike",
            "Alo Yoga",
            "Athleta",
            "Gymshark"
        ],
        "silhouettes": [
            "Body-hugging",
            "Streamlined",
            "Layered",
            "Performance fit",
            "Flexible"
        ]
    },
    "Club Siren": {
        "style_elements": [
            "Sequins",
            "Bodycon fits",
            "Cut-outs",
            "Metallic fabrics",
            "High heels",
            "Statement jewelry",
            "Mini lengths"
        ],
        "color_palette": [
            "#000000",  # Black
            "#FFD700",  # Gold
            "#FF69B4",  # Hot Pink
            "#C0C0C0",  # Silver
            "#FF0000"   # Red
        ],
        "key_brands": [
            "Balmain",
            "House of CB",
            "Oh Polly",
            "Fashion Nova",
            "Pretty Little Thing"
        ],
        "silhouettes": [
            "Bodycon",
            "Mini dress",
            "Cropped",
            "Figure-hugging",
            "Revealing cuts"
        ]
    },
    "Dark Romance": {
        "style_elements": [
            "Lace details",
            "Velvet fabrics",
            "Victorian influences",
            "Gothic accessories",
            "Corset details",
            "Dramatic sleeves",
            "Dark florals"
        ],
        "color_palette": [
            "#000000",  # Black
            "#800020",  # Burgundy
            "#4B0082",  # Indigo
            "#2F4F4F",  # Dark Slate
            "#8B0000"   # Dark Red
        ],
        "key_brands": [
            "The Vampire's Wife",
            "Simone Rocha",
            "Alexander McQueen",
            "Rodarte",
            "Dolce & Gabbana"
        ],
        "silhouettes": [
            "Victorian",
            "Fitted bodice",
            "Full skirts",
            "High necks",
            "Dramatic shapes"
        ]
    },
    "Techwear": {
        "style_elements": [
            "Technical fabrics",
            "Utility pockets",
            "Weather resistance",
            "Modular design",
            "Tactical details",
            "Functional accessories",
            "Minimalist aesthetics"
        ],
        "color_palette": [
            "#000000",  # Black
            "#808080",  # Gray
            "#FFFFFF",  # White
            "#36454F",  # Charcoal
            "#4A4A4A"   # Dark Gray
        ],
        "key_brands": [
            "ACRONYM",
            "Stone Island",
            "Nike ACG",
            "Arc'teryx",
            "Y-3"
        ],
        "silhouettes": [
            "Utilitarian",
            "Modular",
            "Tapered",
            "Articulated",
            "Functional"
        ]
    },
    "Indie Sleaze": {
        "style_elements": [
            "Vintage t-shirts",
            "Ripped tights",
            "Doc Martens",
            "Oversized sweaters",
            "American Apparel",
            "Polaroid aesthetic",
            "Messy hair"
        ],
        "color_palette": [
            "#000000",  # Black
            "#4A4A4A",  # Dark Gray
            "#8B4513",  # Saddle Brown
            "#800020",  # Burgundy
            "#4B0082"   # Indigo
        ],
        "key_brands": [
            "American Apparel",
            "Urban Outfitters",
            "Dr. Martens",
            "Cheap Monday",
            "Jeffrey Campbell"
        ],
        "silhouettes": [
            "Disheveled",
            "Layered",
            "Body-conscious",
            "Mixed proportions",
            "Vintage inspired"
        ]
    },
    "Office Siren": {
        "style_elements": [
            "Tailored blazers",
            "Pencil skirts",
            "Silk blouses",
            "High heels",
            "Statement jewelry",
            "Designer bags",
            "Power suits"
        ],
        "color_palette": [
            "#000000",  # Black
            "#FFFFFF",  # White
            "#1B365D",  # Navy
            "#8B0000",  # Dark Red
            "#808080"   # Gray
        ],
        "key_brands": [
            "Theory",
            "Hugo Boss",
            "Max Mara",
            "Reiss",
            "Equipment"
        ],
        "silhouettes": [
            "Tailored",
            "Fitted",
            "Structured",
            "Professional",
            "Polished"
        ]
    },
    "Regencycore": {
        "style_elements": [
            "Empire waistlines",
            "Puff sleeves",
            "Pearl details",
            "Floral prints",
            "Ribbon trims",
            "Lace accents",
            "Romantic details"
        ],
        "color_palette": [
            "#F5E6E8",  # Light Pink
            "#E8DCC4",  # Cream
            "#87CEEB",  # Sky Blue
            "#E6E6FA",  # Lavender
            "#98FB98"   # Pale Green
        ],
        "key_brands": [
            "Selkie",
            "Hill House Home",
            "Simone Rocha",
            "Zimmermann",
            "Sister Jane"
        ],
        "silhouettes": [
            "Empire waist",
            "A-line",
            "Puff sleeves",
            "High waisted",
            "Floor length"
        ]
    }
}