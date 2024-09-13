import re
import csv
import argparse
import phonenumbers
from datetime import datetime
from collections import defaultdict
from phonenumbers import geocoder

# Function to parse the text file
def parse_data(text, default_region='US'):
    sections = text.split('---')  # Split sections by '---'
    data = {}
    
    for section in sections:
        lines = section.strip().splitlines()
        current_area_code = None
        
        # Check the first line of the section for an area code or multiple area codes
        if lines and re.match(r'^\d{3}(&\d{3})*$', lines[0].strip()):
            current_area_code = lines[0].strip().split('&')
            lines = lines[1:]  # Remove the area code line from processing

        for line in lines:
            # Extract the date if present
            date_match = re.search(r'\((\d{1,2}/\d{1,2})\)', line)
            if date_match:
                date = date_match.group(1)
                try:
                    parsed_date = datetime.strptime(date, "%m/%d").replace(year=datetime.now().year)
                except ValueError:
                    parsed_date = datetime.now()
            else:
                parsed_date = datetime.now()

            # Find and extract phone numbers
            phone_match = re.search(r'\b(\d{3}-\d{3}-\d{4}|\d{3}-\d{4})\b', line)
            if phone_match:
                phone = phone_match.group(0)
                if '-' in phone and len(phone.split('-')[0]) == 3 and len(phone) == 8:  # Handle numbers without area code
                    if current_area_code:
                        phone = f"{current_area_code[0]}-{phone}"
                    else:
                        continue  # Skip if area code is not defined

                # Preserve the full description including the phone number
                description = line.strip()

                # Parse phone number using phonenumbers
                try:
                    parsed_number = phonenumbers.parse(phone, default_region)
                    location = geocoder.description_for_number(parsed_number, "en")
                except phonenumbers.phonenumberutil.NumberParseException:
                    location = "Unknown Location"

                # De-duplicate by checking if the phone number already exists
                if phone not in data:
                    data[phone] = (description, parsed_date.strftime("%Y-%m-%d"), location)

    return list(data.items())

# Function to sort and write to CSV
def write_to_csv(data, output_file):
    # Sort the data based on the phone number
    sorted_data = sorted(data, key=lambda x: x[0])
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Phone Number', 'Description', 'Date', 'Location'])
        
        for phone, (description, date, location) in sorted_data:
            writer.writerow([phone, description, date, location])

# Main function with argparse
def main():
    parser = argparse.ArgumentParser(description='Process a text file of phone numbers and descriptions into a sorted, de-duplicated CSV.')
    parser.add_argument('input_file', help='The input text file containing the phone numbers and descriptions.')
    parser.add_argument('output_file', help='The output CSV file where the processed data will be saved.')
    
    args = parser.parse_args()
    
    # Read the input text file
    with open(args.input_file, 'r') as file:
        text_data = file.read()
    
    # Parse the data
    parsed_data = parse_data(text_data)
    
    # Write to CSV
    write_to_csv(parsed_data, args.output_file)

if __name__ == '__main__':
    main()
