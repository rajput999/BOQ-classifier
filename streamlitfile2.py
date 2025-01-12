import streamlit as st
import pandas as pd
import json
import openai
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelStructureConverter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key

    def read_excel_file(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        """
        Read a single sheet from an Excel file and clean the data.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Excel file not found at: {file_path}")
                
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            logger.info(f"Raw data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df.columns = [str(col).strip() for col in df.columns]
            
            for col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x.strip())
                df[col] = df[col].replace('nan', '')
            
            logger.info(f"Successfully read and cleaned sheet: {sheet_name}")
            logger.info(f"Cleaned data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            st.error(f"Error reading Excel file: {str(e)}")
            raise

    def verify_and_clean_json(self, structured_data: Dict) -> Dict:
        """
        Clean structured data and verify its format
        """
        try:
            logger.info("Starting data cleanup process")
            cleaned_data = structured_data.copy()
            
            sheet_name = list(cleaned_data.keys())[0]
            
            rooms_to_remove = []
            for room_idx, room in enumerate(cleaned_data[sheet_name]):
                if not room.get("Units"):
                    rooms_to_remove.append(room_idx)
                    continue
                    
                units_to_remove = []
                for unit_idx, unit in enumerate(room["Units"]):
                    if not unit.get("Components"):
                        units_to_remove.append(unit_idx)
                        continue
                        
                    components_to_remove = []
                    for comp_idx, component in enumerate(unit["Components"]):
                        if not component:
                            components_to_remove.append(comp_idx)
                            continue
                            
                        non_empty_values = sum(
                            1 for key, value in component.items()
                            if key != "description" and value and str(value).strip()
                        )
                        
                        if non_empty_values <= 1:
                            components_to_remove.append(comp_idx)
                    
                    for idx in reversed(components_to_remove):
                        unit["Components"].pop(idx)
                    
                    if not unit["Components"]:
                        units_to_remove.append(unit_idx)
                
                for idx in reversed(units_to_remove):
                    room["Units"].pop(idx)
                
                if not room["Units"]:
                    rooms_to_remove.append(room_idx)
            
            for idx in reversed(rooms_to_remove):
                cleaned_data[sheet_name].pop(idx)
            
            logger.info(f"Cleanup complete: Removed {len(rooms_to_remove)} empty rooms")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {str(e)}")
            st.error(f"Error during data cleanup: {str(e)}")
            raise

    def is_heading_row(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Enhanced heading detection
        """
        row = row.apply(lambda x: '' if pd.isna(x) else str(x).strip())
        non_empty_cells = [(i, val) for i, val in enumerate(row) if val != '']
        
        if not non_empty_cells:
            return False, ""
        
        total_cells = len(row)
        non_empty_count = len(non_empty_cells)
        empty_ratio = (total_cells - non_empty_count) / total_cells
        contents = [val for _, val in non_empty_cells]
        
        heading_candidates = []
        for pos, content in non_empty_cells:
            if (len(content) <= 3 or 
                re.search(r'^\d', content) or
                re.search(r'\d+\s*(?:mm|cm|m|sqm)', content.lower()) or
                re.search(r'^\d+\.?\d*$', content)):
                continue
                
            empty_before = sum(1 for x in row.iloc[:pos] if x == '')
            word_count = len(content.split())
            if (empty_before > 0 or len(content) > 3) and word_count < 8:
                heading_candidates.append((pos, content, empty_before))
        
        is_heading = empty_ratio > 0.65 and heading_candidates
        
        if is_heading and heading_candidates:
            heading_candidates.sort(key=lambda x: (-x[2], x[0]))
            _, content, _ = heading_candidates[0]
            return True, content
            
        return False, ""

    def extract_json_from_response(self, response_content: str) -> dict:
        """
        Extract JSON from OpenAI response
        """
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
                    st.error(f"Failed to parse extracted JSON: {e}")
                    raise
            else:
                logger.error("No JSON object found in response")
                st.error("No JSON object found in response")
                raise ValueError("No valid JSON found in response")

    def analyze_columns(self, df: pd.DataFrame, sample_rows: int = 5) -> Tuple[List[str], pd.DataFrame, Dict]:
        """
        Analyze and classify columns
        """
        try:
            initial_row_count = len(df)
            empty_row_mask = df.apply(lambda row: all(str(cell).strip() == '' for cell in row), axis=1)
            df = df[~empty_row_mask].reset_index(drop=True)
            removed_rows = initial_row_count - len(df)
            logger.info(f"Removed {removed_rows} empty rows")

            heading_info = df.apply(self.is_heading_row, axis=1)
            heading_mask = pd.Series([info[0] for info in heading_info], index=df.index)
            heading_contents = pd.Series([info[1] for info in heading_info], index=df.index)
            
            data_rows = df[~heading_mask].head(sample_rows)
            sample_str = data_rows.to_string()
            
            analysis_prompt = f"""
            Analyze these Excel columns and their first {sample_rows} rows of data:

            {sample_str}

            Task: Extract and classify the columns based on their relevance to the following categories:

            divide these columns {df.columns.tolist()} into these categories these should be no other column you must be very accurate:

            Important Columns: columns containing these information are required for analysis:
                "description": Detailed description of the item (if applicable)
                "quantity": Value (if applicable)
                "rate": Value (if applicable)
                "unit": Value (if applicable)
                "length": Value (optional)
                "breadth": Value (optional)
                "height": Value (optional)
            Unnecessary Columns: Any columns not matching the categories above.

            THESE JSON VALUES MUST CONTAIN the header row and column names in the DataFrame for these values. Sometimes the actual column name might be "Unnamed":
            RETURN ONLY THE FOLLOWING JSON STRUCTURE WITH NO OTHER TEXT:
            {{
                "Important Columns": {{
                    "description": "<column_name_containing_description>",
                    "quantity": "<column_name_containing_quantity>",
                    "rate": "<column_name_containing_rate>",
                    "unit": "<column_name_containing_unit>",
                    "length": "<column_name_containing_length>",
                    "breadth": "<column_name_containing_breadth>",
                    "height": "<column_name_containing_height>"
                }},
                "Unnecessary Columns": {{
                    "<column_name_1>",
                    "<column_name_2>"
                }}
            }}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Return only the JSON structure with no additional text."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1
            )

            classification = self.extract_json_from_response(
                response['choices'][0]['message']['content'].strip()
            )
            
            important_columns = []
            for category, column in classification["Important Columns"].items():
                if column and column.strip() != f"<column_name_containing_{category}>":
                    important_columns.append(column)
            
            important_columns = list(dict.fromkeys(important_columns))
            valid_columns = [col for col in important_columns if col in df.columns]
            
            heading_column = "Section_Heading/subheading"
            valid_columns = [heading_column] + valid_columns
            
            filtered_df = pd.DataFrame(index=df.index, columns=valid_columns).astype('object')
            
            for idx in df.index:
                if heading_mask[idx]:
                    filtered_df.at[idx, heading_column] = str(heading_contents[idx])
                else:
                    for col in valid_columns[1:]:
                        val = df.at[idx, col]
                        if pd.notna(val):
                            filtered_df.at[idx, col] = str(val) if isinstance(val, (list, tuple)) else val
            
            filtered_df = filtered_df.dropna(how='all')
            
            return valid_columns, filtered_df, classification
            
        except Exception as e:
            logger.error(f"Error during column analysis: {str(e)}")
            st.error(f"Error during column analysis: {str(e)}")
            raise

    def create_description_mapping(self, df: pd.DataFrame, classification: Dict) -> Dict[str, Dict]:
        """
        Create mapping of descriptions to data
        """
        try:
            desc_mapping = {}
            current_section = 0
            
            desc_col = classification["Important Columns"]["description"]
            columns_to_track = {
                "quantity": classification["Important Columns"].get("quantity"),
                "rate": classification["Important Columns"].get("rate"),
                "unit": classification["Important Columns"].get("unit"),
            }
            
            columns_to_track = {k: v for k, v in columns_to_track.items() if v is not None}
            
            for idx, row in df.iterrows():
                if pd.isna(row[desc_col]) or str(row[desc_col]).strip() == '':
                    continue
                
                description = str(row[desc_col]).strip()
                
                if description.upper() in ['SUB TOTAL', 'TOTAL', 'SUBTOTAL']:
                    continue
                
                unique_id = f"{current_section}|{description}"
                
                data = {
                    "metadata": {}
                }
                
                for key, col in columns_to_track.items():
                    if pd.notna(row[col]) and str(row[col]).strip() != '':
                        data["metadata"][key] = str(row[col]).strip()
                
                if data["metadata"]:
                    desc_mapping[unique_id] = data
                
                current_section += 1
            
            logger.info(f"Created mapping for {len(desc_mapping)} descriptions")
            return desc_mapping
                
        except Exception as e:
            logger.error(f"Error creating description mapping: {str(e)}")
            st.error(f"Error creating description mapping: {str(e)}")
            raise

    def add_empty_components(self, data: Dict) -> Dict:
        """
        Add empty components to the structure
        """
        try:
            sheet_name = list(data.keys())[0]
            
            for room in data[sheet_name]:
                for unit in room.get("Units", []):
                    unit["Drawings"] = []
                    
                    for component in unit.get("Components", []):
                        component["Remarks"] = ""
            
            logger.info("Successfully added empty components")
            return data
            
        except Exception as e:
            logger.error(f"Error adding empty components: {str(e)}")
            st.error(f"Error adding empty components: {str(e)}")
            raise

    def add_descriptions_to_components(self, structured_data: Dict, desc_mapping: Dict[str, Dict], sheet_name: str) -> Dict:
        """
        Add descriptions to components
        """
        try:
            remaining_descriptions = desc_mapping.copy()
            
            for room in structured_data[sheet_name]:
                for unit in room.get("Units", []):
                    updated_components = []
                    
                    for component in unit.get("Components", []):
                        if not component:
                            continue
                            
                        matched_desc = None
                        matched_data = None
                        
                        comp_values = {
                            'quantity': str(component.get('quantity', '')).strip(),
                            'rate': str(component.get('rate', '')).strip(),
                            'unit': str(component.get('unit', '')).strip()
                        }
                        
                        for desc, data in list(remaining_descriptions.items()):
                            metadata = data.get('metadata', {})
                            
                            is_match = True
                            for key in ['quantity', 'rate', 'unit']:
                                meta_val = str(metadata.get(key, '')).strip()
                                comp_val = comp_values.get(key, '')
                                try:
                                    if meta_val and comp_val:
                                        if abs(float(meta_val) - float(comp_val)) > 0.01:
                                            is_match = False
                                            break
                                except ValueError:
                                    if meta_val and comp_val and meta_val != comp_val:
                                        is_match = False
                                        break
                            
                            if is_match:
                                matched_desc = desc
                                matched_data = data
                                break
                        
                        if matched_desc and matched_data:
                            updated_component = {
                                "description": matched_desc.split("|")[-1].strip(),
                                "Remarks": "",
                                **component
                            }
                            remaining_descriptions.pop(matched_desc, None)
                        else:
                            updated_component = component
                        
                        updated_components.append(updated_component)
                    
                    unit["Components"] = updated_components
                    unit["Drawings"] = []
            
            total_desc = len(desc_mapping)
            unused_desc = len(remaining_descriptions)
            used_desc = total_desc - unused_desc
            
            logger.info(f"Description addition completed: {used_desc}/{total_desc} descriptions used")
            if unused_desc > 0:
                logger.warning(f"{unused_desc} descriptions were not matched")
                
            return structured_data
            
        except Exception as e:
            logger.error(f"Error adding descriptions to components: {str(e)}")
            st.error(f"Error adding descriptions to components: {str(e)}")
            return structured_data

    def process_sheet_with_openai(self, df: pd.DataFrame, sheet_name: str, desc_mapping: Dict[str, Dict]) -> Dict:
        """
        Process sheet data using OpenAI API
        """
        try:
            data_str = df.to_string()
            prompt = f"""
                You are analyzing unstructured data from an Excel sheet named '{sheet_name}'.  
                Your task is to organize this data into a meaningful three-layer hierarchical JSON format using the following structure:  
                IMPORTANT:  
                - for each heading do not create seperate room , try to group them in subheadings if possible
                - If there are two consecutive none NaN headings in the "section_heading/subheading" column, treat the first as a heading(Room) and the second as a subheading(Unit Name).  
                - Be very accurate about identifying and categorizing headings and subheadings.

                "{sheet_name}": [  
                    {{
                        "Room": "Primary category or theme of the data",  
                        "Units": [  
                            {{
                                "Unit Name": "Subcategories or specific aspects related to the heading",  
                                "Components": [  
                                    {{
                                        "quantity": "Value (if applicable)",  
                                        "rate": "Value (if applicable)",  
                                        "unit": "Value (if applicable)",  
                                        "length": "Value (optional)",  
                                        "breadth": "Value (optional)",  
                                        "height": "Value (optional)"  
                                    }}  
                                ]  
                            }}  
                        ]  
                    }}  
                ]  

                ### Input Data:  
                {data_str}  

                ### Output Format:  
                Return only the structured JSON object strictly following the format. Do not include additional explanations or text.  
                you must maintain the continuty

                IMPORTANT:
                    - Must maintain the three-layer hierarchical
                    - do not left any data , write the full JSON structure be carefull about the last data also
                    - if you do not have information about subheading(unit name) then use the same name as heading
                """

            with st.spinner("Processing data with OpenAI..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a cost analysis expert who creates precise JSON structures."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )

            response_content = response['choices'][0]['message']['content'].strip()
            structured_data = self.extract_json_from_response(response_content)
            return structured_data

        except Exception as e:
            logger.error(f"Error processing sheet with OpenAI: {str(e)}")
            st.error(f"Error processing sheet with OpenAI: {str(e)}")
            raise

    def process_single_sheet(self, input_path: str, sheet_name: str) -> Dict:
        """
        Process a single sheet and return structured data
        """
        try:
            logger.info(f"Starting to process sheet '{sheet_name}' from {input_path}")
            
            # Read and clean the data
            df = self.read_excel_file(input_path, sheet_name)
            
            try:
                # First try processing as small sheet
                data_str = df.to_string()
                prompt = f"""
                    You are analyzing unstructured data from an Excel sheet named '{sheet_name}'.  
                    Your task is to organize this data into a meaningful three-layer hierarchical JSON format using the following structure:  

                    IMPORTANT: There must be exactly ONE room entry that serves as the main category. 

                    "{sheet_name}": [  
                        {{
                            "Room": "Primary category or theme of the data",  
                            "Units": [  
                                {{
                                    "Unit Name": "Subcategories or specific aspects related to the heading",  
                                    "Components": [  
                                        {{
                                            "description": "Detailed description of the item (if applicable)",  
                                            "quantity": "Value (if applicable)",  
                                            "rate": "Value (if applicable)",  
                                            "unit": "Value (if applicable)",  
                                            "length": "Value (optional)",  
                                            "breadth": "Value (optional)",  
                                            "height": "Value (optional)"  
                                        }}  
                                    ]  
                                }}  
                            ]  
                        }}  
                    ]  

                    ### Input Data:  
                    {data_str}  

                    ### Output Format:  
                    Return only the structured JSON object strictly following the format. Do not include additional explanations or text.  
                """

                with st.spinner("Processing small sheet format..."):
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a cost analysis expert who creates precise JSON structures."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )

                structured_data = self.extract_json_from_response(response['choices'][0]['message']['content'].strip())
                structured_data = self.verify_and_clean_json(structured_data)
                structured_data = self.add_empty_components(structured_data)
                
                logger.info("Successfully processed as small sheet")
                return structured_data
                
            except Exception as small_sheet_error:
                logger.warning(f"Small sheet processing failed: {small_sheet_error}")
                st.warning("Attempting large sheet processing method...")
                
                # Try processing as large sheet
                important_columns, filtered_df, classification = self.analyze_columns(df)
                desc_mapping = self.create_description_mapping(filtered_df, classification)
                
                desc_col = classification["Important Columns"]['description']
                filtered_df = filtered_df.drop(columns=[desc_col])
                
                structured_data = self.process_sheet_with_openai(filtered_df, sheet_name, desc_mapping)
                structured_data = self.add_descriptions_to_components(structured_data, desc_mapping, sheet_name)
                structured_data = self.verify_and_clean_json(structured_data)
                
                logger.info("Successfully processed as large sheet")
                return structured_data
            
        except Exception as e:
            logger.error(f"Error processing sheet: {str(e)}")
            st.error(f"Error processing sheet: {str(e)}")
            raise

def main():
    st.title("Excel Structure Converter")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with st.spinner("Reading Excel file..."):
            temp_path = "temp.xlsx"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Get available sheets
        xl = pd.ExcelFile(temp_path)
        sheet_name = st.selectbox("Select sheet", xl.sheet_names)
        
        # API key input
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
        if st.button("Process Sheet"):
            if not api_key:
                st.error("Please enter an API key")
            else:
                try:
                    with st.spinner("Processing..."):
                        converter = ExcelStructureConverter(api_key)
                        structured_data = converter.process_single_sheet(temp_path, sheet_name)
                        
                        # Display the results
                        st.success("Processing completed!")
                        
                        # Show JSON preview
                        st.subheader("Preview of Structured Data")
                        st.json(structured_data)
                        
                        # Download button
                        json_str = json.dumps(structured_data, indent=4)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{sheet_name}_{timestamp}.json"
                        
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=filename,
                            mime="application/json"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Processing error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

if __name__ == "__main__":
    main()