#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import re
import sys
from io import BytesIO

import pandas as pd
from openpyxl import Workbook, load_workbook
from PIL import Image

from common.text_quality import is_gibberish, filter_gibberish
from rag.nlp import find_codec

# copied from `/openpyxl/cell/cell.py`
ILLEGAL_CHARACTERS_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


class RAGFlowExcelParser:
    @staticmethod
    def _load_excel_to_workbook(file_like_object):
        if isinstance(file_like_object, bytes):
            file_like_object = BytesIO(file_like_object)

        # Read first 4 bytes to determine file type
        file_like_object.seek(0)
        file_head = file_like_object.read(4)
        file_like_object.seek(0)

        if not (file_head.startswith(b"PK\x03\x04") or file_head.startswith(b"\xd0\xcf\x11\xe0")):
            logging.info("Not an Excel file, converting CSV to Excel Workbook")

            try:
                file_like_object.seek(0)
                df = pd.read_csv(file_like_object, on_bad_lines='skip')
                return RAGFlowExcelParser._dataframe_to_workbook(df)

            except Exception as e_csv:
                raise Exception(f"Failed to parse CSV and convert to Excel Workbook: {e_csv}")

        try:
            return load_workbook(file_like_object, data_only=True)
        except Exception as e:
            logging.info(f"openpyxl load error: {e}, try pandas instead")
            try:
                file_like_object.seek(0)
                try:
                    dfs = pd.read_excel(file_like_object, sheet_name=None)
                    return RAGFlowExcelParser._dataframe_to_workbook(dfs)
                except Exception as ex:
                    logging.info(f"pandas with default engine load error: {ex}, try calamine instead")
                    file_like_object.seek(0)
                    df = pd.read_excel(file_like_object, engine="calamine")
                    return RAGFlowExcelParser._dataframe_to_workbook(df)
            except Exception as e_pandas:
                raise Exception(f"pandas.read_excel error: {e_pandas}, original openpyxl error: {e}")

    @staticmethod
    def _clean_dataframe(df):
        def clean_string(s):
            if isinstance(s, str):
                return ILLEGAL_CHARACTERS_RE.sub(" ", s)
            return s

        if isinstance(df, dict):
            # Handle case where df is a dict of DataFrames (multiple sheets)
            for sheet_name, sheet_df in df.items():
                if isinstance(sheet_df, pd.DataFrame):
                    # 使用向量化操作替代apply，提升性能
                    df[sheet_name] = sheet_df.applymap(clean_string)
            return df
        elif isinstance(df, pd.DataFrame):
            # Handle case where df is a single DataFrame
            return df.applymap(clean_string)
        else:
            # Return as-is if not a DataFrame or dict
            return df

    @staticmethod
    def _dataframe_to_workbook(df):
        # Clean the dataframe first
        cleaned_df = RAGFlowExcelParser._clean_dataframe(df)
        
        # Handle dict case (multiple sheets)
        if isinstance(cleaned_df, dict):
            if len(cleaned_df) > 1:
                return RAGFlowExcelParser._dataframes_to_workbook(cleaned_df)
            else:
                # Handle single sheet case
                sheet_name = next(iter(cleaned_df.keys()))
                cleaned_df = cleaned_df[sheet_name]

        # Ensure cleaned_df is a DataFrame before accessing columns
        if not isinstance(cleaned_df, pd.DataFrame):
            raise Exception(f"Expected DataFrame, got {type(cleaned_df)}")

        wb = Workbook()
        ws = wb.active
        ws.title = "Data"

        for col_num, column_name in enumerate(cleaned_df.columns, 1):
            ws.cell(row=1, column=col_num, value=column_name)

        for row_num, row in enumerate(cleaned_df.values, 2):
            for col_num, value in enumerate(row, 1):
                ws.cell(row=row_num, column=col_num, value=value)

        return wb
    
    @staticmethod
    def _dataframes_to_workbook(dfs: dict):
        wb = Workbook()
        default_sheet = wb.active
        wb.remove(default_sheet)
        
        for sheet_name, df in dfs.items():
            df = RAGFlowExcelParser._clean_dataframe(df)
            ws = wb.create_sheet(title=sheet_name)
            for col_num, column_name in enumerate(df.columns, 1):
                ws.cell(row=1, column=col_num, value=column_name)
            for row_num, row in enumerate(df.values, 2):
                for col_num, value in enumerate(row, 1):
                    ws.cell(row=row_num, column=col_num, value=value)
        return wb

    @staticmethod
    def _extract_images_from_worksheet(ws, sheetname=None):
        """
        Extract images from a worksheet and enrich them with vision-based descriptions.

        Returns: List[dict]
        """
        images = getattr(ws, "_images", [])
        if not images:
            return []

        raw_items = []

        for img in images:
            try:
                # 限制处理的图片数量，避免内存溢出
                if len(raw_items) >= 20:
                    logging.warning(f"Too many images in worksheet {sheetname}, limiting to 20 images")
                    break
                
                img_bytes = img._data()
                # 限制图片大小，避免处理过大的图片
                if len(img_bytes) > 10 * 1024 * 1024:  # 10MB
                    logging.warning(f"Image too large ({len(img_bytes)/1024/1024:.2f}MB), skipping")
                    continue
                
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                
                # 调整图片大小，减少内存使用
                max_size = 1024
                if pil_img.width > max_size or pil_img.height > max_size:
                    pil_img.thumbnail((max_size, max_size))

                anchor = img.anchor
                if hasattr(anchor, "_from") and hasattr(anchor, "_to"):
                    r1, c1 = anchor._from.row + 1, anchor._from.col + 1
                    r2, c2 = anchor._to.row + 1, anchor._to.col + 1
                    if r1 == r2 and c1 == c2:
                        span = "single_cell"
                    else:
                        span = "multi_cell"
                else:
                    r1, c1 = anchor._from.row + 1, anchor._from.col + 1
                    r2, c2 = r1, c1
                    span = "single_cell"

                item = {
                    "sheet": sheetname or ws.title,
                    "image": pil_img,
                    "image_description": "",
                    "row_from": r1,
                    "col_from": c1,
                    "row_to": r2,
                    "col_to": c2,
                    "span_type": span,
                }
                raw_items.append(item)
            except Exception as e:
                logging.warning(f"Error extracting image: {e}")
                continue
        return raw_items

    def html(self, fnm, chunk_rows=256):
        import gc
        import psutil
        from timeit import default_timer as timer
        
        from html import escape

        file_like_object = BytesIO(fnm) if not isinstance(fnm, str) else fnm
        
        # Log file size
        if isinstance(fnm, bytes):
            logging.info(f"Excel file size: {len(fnm)/1024/1024:.2f} MB")
        
        # Load Excel workbook
        st = timer()
        wb = RAGFlowExcelParser._load_excel_to_workbook(file_like_object)
        logging.info(f"Excel workbook loaded in {timer() - st:.2f}s")
        
        # Log number of sheets
        logging.info(f"Number of sheets: {len(wb.sheetnames)}")
        
        tb_chunks = []

        def _fmt(v):
            if v is None:
                return ""
            return str(v).strip()

        for sheetname in wb.sheetnames:
            logging.info(f"Processing sheet: {sheetname}")
            ws = wb[sheetname]
            
            # Check memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024 /.1024
            logging.info(f"Memory usage before processing sheet {sheetname}: {memory_usage:.2f} GB")
            
            try:
                # Get header row without loading all rows
                header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=False))
                logging.info(f"Header row found with {len(header_row)} columns")
                
                tb_rows_0 = "<tr>"
                for t in header_row:
                    tb_rows_0 += f"<th>{escape(_fmt(t.value))}</th>"
                tb_rows_0 += "</tr>"
                
                # Process data rows in chunks using iter_rows
                row_count = ws.max_row
                logging.info(f"Sheet {sheetname} has {row_count} rows")
                
                if row_count <= 1:
                    logging.info(f"Sheet {sheetname} has no data rows, skipping")
                    continue
                
                # Extract images from worksheet
                st_img = timer()
                images = self._extract_images_from_worksheet(ws, sheetname)
                if images:
                    logging.info(f"Extracted {len(images)} images from sheet {sheetname} in {timer() - st_img:.2f}s")
                
                # Process data rows in chunks
                st_data = timer()
                for chunk_start in range(2, row_count + 1, chunk_rows):
                    chunk_end = min(chunk_start + chunk_rows - 1, row_count)
                    #logging.info(f"Processing rows {chunk_start} to {chunk_end} in sheet {sheetname}")
                    
                    tb = ""
                    tb += f"<table><caption>{sheetname}</caption>"
                    tb += tb_rows_0
                    
                    # Iterate over the current chunk only
                    for r in ws.iter_rows(min_row=chunk_start, max_row=chunk_end, values_only=False):
                        tb += "<tr>"
                        for i, c in enumerate(r):
                            if i < len(header_row):
                                if c.value is None:
                                    tb += "<td></td>"
                                else:
                                    tb += f"<td>{escape(_fmt(c.value))}</td>"
                            else:
                                tb += "<td></td>"
                        tb += "</tr>"
                    tb += "</table>\n"
                    tb_chunks.append(tb)
                    
                    # 每处理完一个chunk就进行垃圾回收，减少内存占用
                    if len(tb_chunks) % 10 == 0:
                        gc.collect()
                        
                logging.info(f"Processed data rows in sheet {sheetname} in {timer() - st_data:.2f}s")
                
                # Add image references to chunks
                if images:
                    for img in images:
                        img_desc = f"Image in {img['sheet']} at position ({img['row_from']},{img['col_from']})"
                        tb_chunks.append(f"<div>{img_desc}</div>\n")
                
                # Check memory usage after processing sheet
                # memory_usage_after = process.memory_info().rss / 1024 / 1024 / 1024
                # logging.info(f"Memory usage after processing sheet {sheetname}: {memory_usage_after:.2f} GB")
                # logging.info(f"Memory change: {memory_usage_after - memory_usage:.2f} GB")
                
                # Clean up after processing each sheet
                # gc.collect()
                # memory_usage_after_gc = process.memory_info().rss / 1024 / 1024 / 1024
                # logging.info(f"Memory usage after GC: {memory_usage_after_gc:.2f} GB")
                    
            except Exception as e:
                logging.warning(f"Skip sheet '{sheetname}' due to rows access error: {e}")
                continue
        
        # Final memory check
        # process = psutil.Process()
        # final_memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        # logging.info(f"Final memory usage after parsing Excel: {final_memory_usage:.2f} GB")
        logging.info(f"Total tb_chunks processed: {len(tb_chunks)}")

        return tb_chunks

    def markdown(self, fnm):
        import pandas as pd

        file_like_object = BytesIO(fnm) if not isinstance(fnm, str) else fnm
        try:
            file_like_object.seek(0)
            df = pd.read_excel(file_like_object)
        except Exception as e:
            logging.warning(f"Parse spreadsheet error: {e}, trying to interpret as CSV file")
            file_like_object.seek(0)
            df = pd.read_csv(file_like_object, on_bad_lines='skip')
        df = df.replace(r"^\s*$", "", regex=True)
        return df.to_markdown(index=False)

    def __call__(self, fnm):
        import gc
        import psutil
        from timeit import default_timer as timer
        
        file_like_object = BytesIO(fnm) if not isinstance(fnm, str) else fnm
        
        # Log file size
        if isinstance(fnm, bytes):
            logging.info(f"Excel file size: {len(fnm)/1024/1024:.2f} MB")
        
        # Load Excel workbook
        st = timer()
        wb = RAGFlowExcelParser._load_excel_to_workbook(file_like_object)
        logging.info(f"Excel workbook loaded in {timer() - st:.2f}s")
        
        # Log number of sheets
        logging.info(f"Number of sheets: {len(wb.sheetnames)}")
        
        res = []
        process = psutil.Process()
        
        for sheetname in wb.sheetnames:
            logging.info(f"Processing sheet: {sheetname}")
            ws = wb[sheetname]
            
            # Check memory usage
            memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
            logging.info(f"Memory usage before processing sheet {sheetname}: {memory_usage:.2f} GB")
            
            try:
                # Get header row without loading all rows
                header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=False))
                logging.info(f"Header row found with {len(header_row)} columns")
                ti = [str(t.value) if t.value else "" for t in header_row]
                
                # Process data rows using iter_rows
                row_count = ws.max_row
                logging.info(f"Sheet {sheetname} has {row_count} rows")
                
                if row_count <= 1:
                    logging.info(f"Sheet {sheetname} has no data rows, skipping")
                    continue
                
                # Extract images from worksheet
                st_img = timer()
                images = self._extract_images_from_worksheet(ws, sheetname)
                if images:
                    logging.info(f"Extracted {len(images)} images from sheet {sheetname} in {timer() - st_img:.2f}s")

                
                # Process data rows in chunks
                chunk_size = 1000
                st_data = timer()
                rows_processed = 0
                consecutive_empty_rows = 0
                max_consecutive_empty_rows = 100  # 当遇到100个连续空行时停止处理
                
                for chunk_start in range(2, row_count + 1, chunk_size):
                    chunk_end = min(chunk_start + chunk_size - 1, row_count)
                    # logging.info(f"Processing rows {chunk_start} to {chunk_end} in sheet {sheetname}")
                    
                    # Iterate over the current chunk only
                    for r in ws.iter_rows(min_row=chunk_start, max_row=chunk_end, values_only=False):
                        fields = []
                        has_data = False
                        
                        for i, c in enumerate(r):
                            if c.value:
                                has_data = True
                                if i < len(ti) and ti[i]:
                                    field = f"{ti[i]}：{c.value}"
                                else:
                                    field = str(c.value)
                                fields.append(field)
                        
                        # Check if row has data
                        if has_data:
                            consecutive_empty_rows = 0
                            line = "; ".join(fields)
                            if sheetname.lower().find("sheet") < 0:
                                line += " ——" + sheetname
                            if line:
                                # Filter gibberish content
                                if not is_gibberish(line):
                                    filtered_line = filter_gibberish(line)
                                    if filtered_line:
                                        res.append(filtered_line)
                                        rows_processed += 1
                                else:
                                    logging.debug(f"Filtered gibberish row from Excel: {line[:100]}...")
                                
                                # Check memory usage every 100 rows
                                # if rows_processed % 100 == 0:
                                #     current_memory = process.memory_info().rss / 1024 / 1024 / 1024
                                #     logging.info(f"Processed {rows_processed} rows, memory usage: {current_memory:.2f} GB")
                        else:
                            consecutive_empty_rows += 1
                            # If we've hit too many consecutive empty rows, stop processing
                            if consecutive_empty_rows >= max_consecutive_empty_rows:
                                logging.info(f"Found {consecutive_empty_rows} consecutive empty rows, stopping processing of sheet {sheetname}")
                                break
                    
                    # If we've hit too many consecutive empty rows, stop processing
                    if consecutive_empty_rows >= max_consecutive_empty_rows:
                        break
                
                logging.info(f"Processed {rows_processed} rows in sheet {sheetname} in {timer() - st_data:.2f}s")
                
                # Check memory usage after processing sheet
                # memory_usage_after = process.memory_info().rss / 1024 / 1024 / 1024
                # logging.info(f"Memory usage after processing sheet {sheetname}: {memory_usage_after:.2f} GB")
                # logging.info(f"Memory change: {memory_usage_after - memory_usage:.2f} GB")
                
                # Clean up after processing each sheet
                # gc.collect()
                # memory_usage_after_gc = process.memory_info().rss / 1024 / 1024 / 1024
                # logging.info(f"Memory usage after GC: {memory_usage_after_gc:.2f} GB")
                    
            except Exception as e:
                logging.warning(f"Skip sheet '{sheetname}' due to rows access error: {e}")
                continue
        
        # Final memory check
        # final_memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        # logging.info(f"Final memory usage after parsing Excel: {final_memory_usage:.2f} GB")
        logging.info(f"Total rows processed: {len(res)}")

        return res

    @staticmethod
    def row_number(fnm, binary):
        if fnm.split(".")[-1].lower().find("xls") >= 0:
            wb = RAGFlowExcelParser._load_excel_to_workbook(BytesIO(binary))
            total = 0
            
            for sheetname in wb.sheetnames:
               try:
                   ws = wb[sheetname]
                   # Use max_row instead of loading all rows to memory
                   total += ws.max_row
               except Exception as e:
                   logging.warning(f"Skip sheet '{sheetname}' due to rows access error: {e}")
                   continue
            return total

        if fnm.split(".")[-1].lower() in ["csv", "txt"]:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
            return len(txt.split("\n"))


if __name__ == "__main__":
    psr = RAGFlowExcelParser()
    psr(sys.argv[1])
