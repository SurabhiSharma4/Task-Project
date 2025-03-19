# AI-Product-Placement
README - AI Product Placement Tool
Project Overview
The AI Product Placement Tool allows users to integrate e-commerce product images into lifestyle images based on user prompts. It uses object detection and natural language processing to position products realistically.
Features
•	Upload any lifestyle and product images.
•	Provide a prompt to describe placement.
•	AI automatically detects surfaces and adjusts position.
•	Resizes product to realistic proportions.
•	Blends product seamlessly into the background.
•	Download the final composite image.
Installation Instructions
Prerequisites
•	Install Python 3.12.0
•	Install necessary dependencies using pip
Requirements.txt file
•	streamlit
•	opencv-python
•	rembg
•	ultralytics
•	numpy
•	transformers 
Steps
1.	Clone the repository:  	git clone <repo_link>
2.	Navigate to the directory:      cd AI-Product-Placement
3.	Create a virtual environment:	python -m venv env
4.	Activate the virtual environment:
o	Windows:    env\Scripts\activate
o	Mac/Linux:     source env/bin/activate
5.	Install dependencies:	pip install -r requirements.txt
6.	Run the Streamlit app:	streamlit run app.py --server.fileWatcherType none
How to Use
1.	Upload a product image (PNG/JPG) and a lifestyle image.
2.	Type a prompt describing where you want the product placed (e.g., "Place the lamp on the right side of the table.").
3.	Click the 'Apply Placement' button to process the image.
4.	The AI will detect surfaces and place the product accordingly.
5.	Download the final composite image.
Example Usage
Input Prompt:
"Place the lamp on the right side of the table."
AI Process:
1.	Detects the table in the lifestyle image.
2.	Adjusts the size of the lamp to look realistic.
3.	Places the lamp on the right side of the table.
4.	Blends the lamp naturally into the background.
Output:
A realistic composite image with the lamp correctly positioned.

