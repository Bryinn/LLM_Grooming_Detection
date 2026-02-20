DEBUG_MODE = False  # Set to True to enable debug output
DEBUG_FILE = "7_Kent_Grant.txt"  # Set to the filename to debug (case-sensitive)
import re

regex_order = [
    # Email from ... : message
    ('email_from', re.compile(r"^Email from ([^:]+):\s*(.+)$", re.IGNORECASE)),
    # Reply from ... : message
    ('reply_from', re.compile(r"^Reply from ([^:]+):\s*(.+)$", re.IGNORECASE)),
    # author (date time) : message  e.g. "user (07/19/09 5:35:43 PM): msg"
    ('user_paren_datetime', re.compile(r"^([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?)\s*\((\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\):\s*(.+)$", re.IGNORECASE)),
    # author (time or time AM/PM) : message  e.g. "user (3:20:16 PM): msg" or "user (3:20:16): msg"
    ('user_paren_timeonly', re.compile(r"^([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?)\s*\((\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM))?)\):\s*(.+)$", re.IGNORECASE)),
    # time-only at start (no username), e.g. "17:16 PM): message" (only valid if last_author is set)
    ('time_only_colon', re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\)):\s*(.+)$", re.IGNORECASE)),
    # [HH:MM] user: message
    ('bracket_time_user_colon', re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?):\s*(.+)$")),
    # username [time]: message (e.g., 'Xcursion24 [3:01 P.M.]: socali here')
    ('user_bracket_time_colon', re.compile(r"^([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+) \[(\d{1,2}:\d{2}(?:\s*[APMapm.]+)?)\]:\s*(.+)$", re.IGNORECASE)),
    # YYYY-MM-DD HH:MM:SS user: message
    ('iso_date_user_colon', re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?):\s*(.+)$")),
    # MM/DD/YYYY HH:MM - user: message
    ('date_time_dash_user_colon', re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?):\s*(.+)$")),
    # user: message [time at end] e.g. "Tory Beltz Phone: Hey 3:02 PM"
    ('user_colon_time_end', re.compile(r"^([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?):\s*(.+?)\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM))$", re.IGNORECASE)),
    # user: message  (fallback without explicit time)
    ('user_colon', re.compile(r"^([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?):\s*(.+)$")),
    # user - message  (only accept as message when left side is not a date)
    ('user_dash', re.compile(r"^([A-Za-z0-9_.@&\-\s'`\[\](){}|/\\,:;!?$#~^%*+]+?)\s*-\s*(.+)$")),
    # quoted lines starting with > or &gt;
    ('quoted_gt', re.compile(r'^[>\u003E\s]*&?gt;?\s*(.+)$')),
]

def parse_messages(joined_lines, regex_order, date_line_re):
    """
    Parse messages from joined_lines using regex_order and date_line_re.
    Returns message_blocks (list of lists of message dicts).
    """
    if DEBUG_MODE and filename == DEBUG_FILE:
        print(f"[DEBUG] Adding message to block: author={author!r}, message_text={message_text!r}, line={file_ln}")
    message_blocks = []
    current_block = []
    last_dt = None
    current_date = None
    last_author = None
    for file_ln, raw_line in joined_lines:
        line = raw_line.strip()
        if not line:
            continue
        # Detect date line and update current_date
        date_match = date_line_re.match(line)
        if date_match:
            current_date = date_match.group(1)
            continue
        matched = False
        for name, rx in regex_order:
            m = rx.match(line)
            if not m:
                continue
            author = None
            raw_dt = None
            message_text = None
            dt_obj = None
            if name == 'iso_date_user_colon':
                raw_dt = f"{m.group(1)} {m.group(2)}"
                author = m.group(3).strip()
                message_text = m.group(4).strip()
                dt_obj = parse_datetime(raw_dt)
                last_author = author
            elif name == 'date_time_dash_user_colon':
                raw_dt = f"{m.group(1)} {m.group(2)}"
                author = m.group(3).strip()
                message_text = m.group(4).strip()
                dt_obj = parse_datetime(raw_dt)
                last_author = author
            elif name == 'user_colon_time_end':
                author = m.group(1).strip()
                message_text = m.group(2).strip()
                time_part = m.group(3).strip()
                if current_date:
                    raw_dt = f"{current_date} {time_part}"
                else:
                    raw_dt = time_part
                dt_obj = parse_datetime(raw_dt)
                last_author = author
            elif name == 'user_colon':
                author = m.group(1).strip()
                message_text = m.group(2).strip()
                last_author = author
            elif name == 'user_dash':
                author = m.group(1).strip()
                message_text = m.group(2).strip()
                last_author = author
            elif name == 'quoted_gt':
                author = None
                message_text = m.group(1).strip()
            # If this is the first message in a block, or if 3-hour gap, start new block
            start_new_block = False
            if not current_block:
                start_new_block = True
            elif dt_obj and last_dt:
                try:
                    delta = abs((dt_obj - last_dt).total_seconds()) / 60.0
                except Exception:
                    delta = None
                if delta is not None and delta > 180:
                    start_new_block = True
            if start_new_block and current_block:
                message_blocks.append(current_block)
                current_block = []
            current_block.append({
                'author': author,
                'text': message_text,
                'datetime': dt_obj,
                'raw_datetime': raw_dt,
                'line_number': file_ln,
                'format': name
            })
            last_dt = dt_obj if dt_obj else last_dt
            matched = True
            break
        # Handle wrapped message lines: if not matched and previous line was a message, append to previous message
        if not matched and current_block and not line.startswith(' '):
            prev_msg = current_block[-1]
            prev_msg['text'] += ' ' + line
            continue
    if current_block:
        message_blocks.append(current_block)
    return message_blocks
def normalize_username(username):
    """
    Normalize usernames for participant extraction. Removes parenthetical/date fragments, filters out context headers, and excludes non-usernames.
    Returns cleaned username or None if not valid.
    """
    if DEBUG_MODE and DEBUG_FILE == '119_Xcursion24.txt':
        print(f"[DEBUG] normalize_username called with: {username!r}")
    if not username:
        return None
    cleaned = username.strip()
    cleaned = re.sub(r'\s*\([^)]*$', '', cleaned)  # Remove open parenthesis at end
    cleaned = re.sub(r'\s*\([^)]*\)$', '', cleaned)  # Remove closed parenthesis at end
    cleaned = re.sub(r'\s*\([^)]*[\d:/APMapm -][^)]*\)?$', '', cleaned)
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\s+\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM)?$', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(':').strip()
    if not cleaned:
        return None
    # Allow more flexible usernames for email/chat logs
    context_clues = [
        r'^Email Exchange(\s|$)', r'^OFFLINE(\s|$)', r'^Text Messaging(\s|$)', r'^Meetme.com(\s|$)',
        r'^Chat Log(\s|$)', r'^Conversation(\s|$)', r'^Session(\s|$)', r'^Log(\s|$)', r'^Message(\s|$)',
        r'^IMs?(\s|$)', r'^AIM(\s|$)', r'^Yahoo(\s|$)', r'^MSN(\s|$)', r'^ICQ(\s|$)', r'^Skype(\s|$)',
        r'^Gmail(\s|$)', r'^Hotmail(\s|$)', r'^Facebook(\s|$)', r'^MySpace(\s|$)', r'^Google(\s|$)',
        r'^Exchange(\s|$)', r'^SMS(\s|$)', r'^Transcript(\s|$)', r'^\d{1,2}/\d{1,2}/\d{2,4}$',
    ]
    for clue in context_clues:
        if re.match(clue, cleaned, re.IGNORECASE):
            return None
    # Allow email addresses and multi-word names as usernames
    if re.match(r'.+?: .+', cleaned):
        return None
    if re.match(r'^\d+$', cleaned):
        return None
    if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', cleaned):
        return None
    # Loosen length and space restrictions for email/chat logs
    if len(cleaned) > 60:
        return None
    if cleaned.count(' ') > 7:
        return None
    # Accept any non-empty cleaned username (including emails, multi-word names)
    if not re.match(r'^[A-Za-z0-9_.@&\- ]+$', cleaned):
        return None
    return cleaned if cleaned else None
def extract_contextual_lines(joined_lines):
    """
    Given joined_lines (list of (line_number, raw_line)),
    group all consecutive parenthetical comments (even if embedded after chat lines)
    into a single contextual block, and flush when a non-parenthetical or empty line is encountered.
    Returns a list of dicts: {'line_number': int, 'text': str}
    """
    contextual_lines = []
    paren_buffer = []
    paren_start_ln = None
    for file_ln, raw_line in joined_lines:
        line = raw_line.strip()
        if not line:
            continue
        parens = re.findall(r'\([^\)]*\)', raw_line)
        if parens:
            if paren_start_ln is None:
                paren_start_ln = file_ln
            paren_buffer.extend([f'({p})' for p in parens])
        else:
            if paren_buffer:
                contextual_lines.append({
                    'line_number': paren_start_ln,
                    'text': ' '.join(paren_buffer).strip()
                })
                paren_buffer = []
                paren_start_ln = None
            if line.strip() and not re.match(r'.*\(.*\).*', raw_line):
                contextual_lines.append({
                    'line_number': file_ln,
                    'text': raw_line
                })
        next_idx = joined_lines.index((file_ln, raw_line)) + 1
        if next_idx >= len(joined_lines) or not joined_lines[next_idx][1].strip():
            if paren_buffer:
                contextual_lines.append({
                    'line_number': paren_start_ln,
                    'text': ' '.join(paren_buffer).strip()
                })
                paren_buffer = []
                paren_start_ln = None
    return contextual_lines
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

def filter_conversations(input_dir="datasets", output_dir="filtered_datasets"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Process subdirectories in the datasets folder
    for subdirectory in os.listdir(input_dir):
        subdirectory_path = os.path.join(input_dir, subdirectory)
        if not os.path.isdir(subdirectory_path):
            continue
        if subdirectory == "PJ":
            process_pj_txt(subdirectory_path, output_dir)
        # ...existing code for other subdirectories...
def parse_datetime(date_time_str):
    """Try to parse a full date+time string into a datetime object.
    Accepts formats like '7/24/14 6:05 PM' or '07/24/2014 18:05'.
    Returns None if unable to parse.
    """
    if not date_time_str:
        return None
    date_time_str = date_time_str.strip()
    # Common formats to try
    fmts = [
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%B %d, %Y %I:%M %p",   # e.g., March 27, 2005 3:01 PM
        "%B %d, %Y %H:%M",      # e.g., March 27, 2005 15:01
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_time_str, fmt)
        except:
            continue
    # As a fallback, try to extract just the time part
    parts = date_time_str.split()
    if parts:
        time_only = ' '.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        try:
            return datetime.strptime(time_only, "%I:%M %p")
        except:
            try:
                return datetime.strptime(time_only, "%H:%M")
            except:
                return None
    return None

def process_pj_txt(directory_path, output_dir):
    """Process PJ text conversation files

    - Process all files (not only .txt)
    - Treat any lines that do not strictly match the format
      "<username> (<date> <time>): <message>" as contextual comments
      and record them in metadata['contextual_lines'] with line numbers.
    - Split messages into conversations when more than 3 hours pass
      between consecutive messages (using parsed datetimes when available).
    """
    pj_output_dir = os.path.join(output_dir, "PJ")
    os.makedirs(pj_output_dir, exist_ok=True)

    total_kept = 0
    total_skipped = 0
    total_conv_skipped = 0

    for filename in os.listdir(directory_path):
        input_path = os.path.join(directory_path, filename)

        # skip directories
        if os.path.isdir(input_path):
            continue

        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Normalize lines (strip trailing newlines only)
            lines = [ln.rstrip('\n') for ln in lines]

            metadata = {}
            metadata['contextual_lines'] = []

            # Extract metadata from the top of the file
            metadata_end_idx = 0
            for idx, raw_line in enumerate(lines):
                line = raw_line.strip()
                if line.startswith('File:'):
                    metadata['file'] = line.split('\t')[-1]
                    metadata_end_idx = idx
                elif line.startswith('Name:'):
                    metadata['name'] = line.split('\t')[-1]
                    metadata_end_idx = idx
                elif line.startswith('Location:'):
                    metadata['location'] = line.split('\t')[-1]
                    metadata_end_idx = idx
                elif line.startswith('Date:'):
                    metadata['date'] = line.split('\t')[-1]
                    metadata_end_idx = idx
                elif line.startswith('Link:'):
                    metadata['link'] = line.split('\t')[-1]
                    metadata_end_idx = idx
                else:
                    # stop metadata collection when a non-metadata non-empty line appears
                    if line != '':
                        break

            # Body lines start after metadata_end_idx
            body_start = metadata_end_idx + 1
            # Store file-level date for fallback
            file_date = metadata.get('date', None)
            # Track the most recent contextual date cue (e.g., 'March 27, 2005' or 01-03-2006)
            contextual_date = None

            # Collect message blocks, using multiple regexes to accept
            # the variety of formats observed in the dataset. Non-matching
            # lines are recorded as contextual lines with their file line numbers.
            import re


            # New logic: split conversations on context clues (contextual lines) or 3-hour gaps
            message_blocks = []
            current_block = []
            last_dt = None
            current_date = None
            last_author = None
            #import re
            date_line_re = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})$')

            # Preprocess: join wrapped lines before parsing
            joined_lines = []
            buffer = ''
            buffer_ln = None
            context_cue_re = re.compile(r'^(Text Messaging|Email Exchange|Meetme.com|Chat Log|Conversation|Session|Log|Message|IMs?|AIM|Yahoo|MSN|ICQ|Skype|Gmail|Hotmail|Facebook|MySpace|Google|Exchange|SMS|Transcript)(\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4})?$|^(Text Messaging|Email Exchange|Meetme.com|Chat Log|Conversation|Session|Log|Message|IMs?|AIM|Yahoo|MSN|ICQ|Skype|Gmail|Hotmail|Facebook|MySpace|Google|Exchange|SMS|Transcript)$', re.IGNORECASE)
            for idx, raw_line in enumerate(lines[body_start:], start=body_start+1):
                line = raw_line.rstrip('\n')
                if not line.strip():
                    if buffer:
                        joined_lines.append((buffer_ln, buffer.strip()))
                        buffer = ''
                        buffer_ln = None
                    continue
                # If line matches context cue pattern, flush buffer and add cue as standalone
                if context_cue_re.match(line.strip()):
                    if buffer:
                        joined_lines.append((buffer_ln, buffer.strip()))
                        buffer = ''
                        buffer_ln = None
                    joined_lines.append((idx, line.strip()))
                    continue
                # If line matches a message pattern, flush buffer
                is_message = False
                for _, rx in regex_order:
                    if rx.match(line.strip()):
                        is_message = True
                        break
                if is_message:
                    if buffer:
                        joined_lines.append((buffer_ln, buffer.strip()))
                    buffer = line
                    buffer_ln = idx
                else:
                    # Wrapped line, append to buffer
                    if buffer:
                        buffer += ' ' + line.strip()
                    else:
                        buffer = line
                        buffer_ln = idx
            if buffer:
                joined_lines.append((buffer_ln, buffer.strip()))


            # Define context cue pattern: starts with known header, optional date
            context_cue_re = re.compile(r'^(Text Messaging|Email Exchange|Meetme.com|Chat Log|Conversation|Session|Log|Message|IMs?|AIM|Yahoo|MSN|ICQ|Skype|Gmail|Hotmail|Facebook|MySpace|Google|Exchange|SMS|Transcript)(\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4})?$|^(Text Messaging|Email Exchange|Meetme.com|Chat Log|Conversation|Session|Log|Message|IMs?|AIM|Yahoo|MSN|ICQ|Skype|Gmail|Hotmail|Facebook|MySpace|Google|Exchange|SMS|Transcript)$', re.IGNORECASE)



            for file_ln, raw_line in joined_lines:
                line = raw_line.strip()
                if DEBUG_MODE and filename == DEBUG_FILE:
                    print(f"[DEBUG] Processing line {file_ln}: '{line}'")
                if not line:
                    continue

                # Detect contextual date cue (e.g., 'March 27, 2005' or '01-03-2006')
                date_cue_match = re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}$', line)
                dash_date_cue_match = re.match(r'^(\d{2})-(\d{2})-(\d{4})$', line)
                if date_cue_match:
                    contextual_date = line.strip()
                    metadata['contextual_lines'].append({
                        'line_number': file_ln,
                        'text': raw_line,
                        'date': contextual_date
                    })
                    # Always split conversation on context cue
                    if current_block:
                        message_blocks.append(current_block)
                        current_block = []
                    last_dt = None
                    last_author = None
                    continue
                elif dash_date_cue_match:
                    # Convert DD-MM-YYYY or MM-DD-YYYY to a parseable format
                    # We'll assume MM-DD-YYYY (US style) for this dataset
                    mm, dd, yyyy = dash_date_cue_match.groups()
                    contextual_date = f"{mm}/{dd}/{yyyy}"
                    metadata['contextual_lines'].append({
                        'line_number': file_ln,
                        'text': raw_line,
                        'date': contextual_date
                    })
                    if current_block:
                        message_blocks.append(current_block)
                        current_block = []
                    last_dt = None
                    last_author = None
                    continue

                # Detect date line and update current_date, skip for participant extraction
                date_match = date_line_re.match(line)
                if date_match:
                    current_date = date_match.group(1)
                    continue

                # Detect context cue lines (e.g., 'Text Messaging 7/19/12')
                cue_match = context_cue_re.match(line)
                if cue_match:
                    cue_text = raw_line
                    # Try to extract date from cue
                    cue_date = None
                    # Look for date in cue (e.g., 'Text Messaging 7/19/12')
                    date_search = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', line)
                    if date_search:
                        cue_date = date_search.group(1)
                        current_date = cue_date
                    else:
                        # Use previous current_date, file_date, or 'N/A'
                        cue_date = current_date if current_date else (file_date if file_date else 'N/A')
                    # Debug output for every detected context cue
                    if DEBUG_MODE and filename == DEBUG_FILE:
                        print(f"[DEBUG] Context cue detected: '{cue_text}' (line {file_ln}) -> date: {cue_date}")
                    metadata['contextual_lines'].append({
                        'line_number': file_ln,
                        'text': cue_text,
                        'date': cue_date
                    })
                    # Always split conversation on context cue
                    if current_block:
                        message_blocks.append(current_block)
                        current_block = []
                    last_dt = None
                    last_author = None
                    continue

                matched = False

                for name, rx in regex_order:
                    m = rx.match(line)
                    if not m:
                        continue

                    # Basic post-match validation to avoid grabbing header/footer lines
                    if name == 'user_colon':
                        left = m.group(1).strip()
                        if left.lower() in ('file', 'name', 'location', 'date', 'link'):
                            matched = False
                            break
                    if name == 'user_dash':
                        left = m.group(1).strip()
                        if re.match(r'^\d{1,2}-\d{1,2}-\d{2,4}$', left) or len(left) > 60:
                            matched = False
                            break

                    author = None
                    raw_dt = None
                    message_text = None
                    dt_obj = None
                    if name == 'email_from' or name == 'reply_from':
                        author = m.group(1).strip()
                        message_text = m.group(2).strip()
                        last_author = author
                    elif name == 'user_paren_datetime':
                        author = m.group(1).strip()
                        dt_str = m.group(2).strip()
                        message_text = m.group(3).strip()
                        raw_dt = dt_str
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'user_paren_timeonly':
                        author = m.group(1).strip()
                        time_part = m.group(2).strip()
                        message_text = m.group(3).strip()
                        # Use contextual_date, current_date, else file_date, else just time
                        if contextual_date:
                            raw_dt = f"{contextual_date} {time_part}"
                        elif current_date:
                            raw_dt = f"{current_date} {time_part}"
                        elif file_date:
                            raw_dt = f"{file_date} {time_part}"
                        else:
                            raw_dt = time_part
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'time_only_colon':
                        # Only treat as valid message if last_author is set
                        if last_author:
                            author = last_author
                            time_part = m.group(1).replace(')', '').strip()
                            message_text = m.group(2).strip()
                            if contextual_date:
                                raw_dt = f"{contextual_date} {time_part}"
                            elif current_date:
                                raw_dt = f"{current_date} {time_part}"
                            else:
                                raw_dt = time_part
                            dt_obj = parse_datetime(raw_dt)
                        else:
                            continue
                    elif name == 'bracket_time_user_colon':
                        time_part = m.group(1).strip()
                        author = m.group(2).strip()
                        message_text = m.group(3).strip()
                        if contextual_date:
                            raw_dt = f"{contextual_date} {time_part}"
                        elif current_date:
                            raw_dt = f"{current_date} {time_part}"
                        else:
                            raw_dt = time_part
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'user_bracket_time_colon':
                        author = m.group(1).strip()
                        time_part = m.group(2).strip()
                        message_text = m.group(3).strip()
                        # Normalize time_part: remove brackets, dots, and fix AM/PM
                        norm_time = time_part.replace('.', '').replace('[', '').replace(']', '').replace('  ', ' ').strip()
                        # If time is like '3:01 PM' or '3:01 P.M.'
                        if 'M' not in norm_time.upper() and len(norm_time.split(':')) == 2:
                            # If no AM/PM, assume 24-hour or missing
                            pass
                        # Combine with contextual_date if present
                        if contextual_date:
                            raw_dt = f"{contextual_date} {norm_time}"
                        elif current_date:
                            raw_dt = f"{current_date} {norm_time}"
                        else:
                            raw_dt = norm_time
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'iso_date_user_colon':
                        raw_dt = f"{m.group(1)} {m.group(2)}"
                        author = m.group(3).strip()
                        message_text = m.group(4).strip()
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'date_time_dash_user_colon':
                        raw_dt = f"{m.group(1)} {m.group(2)}"
                        author = m.group(3).strip()
                        message_text = m.group(4).strip()
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'user_colon_time_end':
                        author = m.group(1).strip()
                        message_text = m.group(2).strip()
                        time_part = m.group(3).strip()
                        if contextual_date:
                            raw_dt = f"{contextual_date} {time_part}"
                        elif current_date:
                            raw_dt = f"{current_date} {time_part}"
                        else:
                            raw_dt = time_part
                        dt_obj = parse_datetime(raw_dt)
                        last_author = author
                    elif name == 'user_colon':
                        author = m.group(1).strip()
                        message_text = m.group(2).strip()
                        last_author = author
                    elif name == 'user_dash':
                        author = m.group(1).strip()
                        message_text = m.group(2).strip()
                        last_author = author
                    elif name == 'quoted_gt':
                        author = None
                        message_text = m.group(1).strip()

                    # Handle parenthetical comments at the end of a message (after joining lines)
                    import re as _re
                    comment_match = _re.search(r'\(([^)]{10,})\)\s*$', message_text)
                    if comment_match:
                        comment = comment_match.group(0)
                        # Remove comment from message text
                        message_text = message_text[:comment_match.start()].rstrip()
                        # Add comment as contextual line
                        metadata['contextual_lines'].append({
                            'line_number': file_ln,
                            'text': comment.strip()
                        })

                    # If this is the first message in a block, or if 3-hour gap, or if previous block ended due to context clue, start new block
                    start_new_block = False
                    if not current_block:
                        start_new_block = True
                    elif dt_obj and last_dt:
                        try:
                            delta = abs((dt_obj - last_dt).total_seconds()) / 60.0
                        except Exception:
                            delta = None
                        if delta is not None and delta > 180:
                            start_new_block = True

                    if start_new_block and current_block:
                        message_blocks.append(current_block)
                        current_block = []

                    current_block.append({
                        'author': author,
                        'text': message_text,
                        'datetime': dt_obj,
                        'raw_datetime': raw_dt,
                        'line_number': file_ln,
                        'format': name
                    })
                    last_dt = dt_obj if dt_obj else last_dt
                    matched = True
                    break

                # Handle wrapped message lines: if not matched and previous line was a message, append to previous message
                if not matched and current_block and not line.startswith(' '):
                    # Only append if the line is not empty and not a new message
                    prev_msg = current_block[-1]
                    prev_msg['text'] += ' ' + line
                    continue

                if not matched:
                    # Merge all parenthetical comments into a single block for each conversation
                    if 'paren_buffer' not in locals():
                        paren_buffer = []
                        paren_start_ln = None
                    # If line is not a message and not empty, check for parenthetical
                    if line.strip():
                        if line.lstrip().startswith('('):
                            if paren_start_ln is None:
                                paren_start_ln = file_ln
                            paren_buffer.append(raw_line)
                        else:
                            # If buffer has parentheticals, flush as one block
                            if paren_buffer:
                                metadata['contextual_lines'].append({
                                    'line_number': paren_start_ln,
                                    'text': '\n'.join(paren_buffer).strip()
                                })
                                paren_buffer = []
                                paren_start_ln = None
                            # Merge narrative sections spanning multiple lines
                            if not re.match(r'.*\(.*\).*', raw_line):
                                # Only add if not a parenthetical
                                metadata['contextual_lines'].append({
                                    'line_number': file_ln,
                                    'text': raw_line
                                })
                    # If next line is empty or end of joined_lines, flush buffer
                    next_idx = joined_lines.index((file_ln, raw_line)) + 1
                    if next_idx >= len(joined_lines) or not joined_lines[next_idx][1].strip():
                        if paren_buffer:
                            metadata['contextual_lines'].append({
                                'line_number': paren_start_ln,
                                'text': '\n'.join(paren_buffer).strip()
                            })
                            paren_buffer = []
                            paren_start_ln = None
                    # Context clue: always start a new conversation block
                    context_split = False
                    if (
                        re.search(r'(Text Messaging|Email Exchange|Meetme.com|Chat Log|Conversation|Session|Log|Message|IMs?|AIM|Yahoo|MSN|ICQ|Skype|Gmail|Hotmail|Facebook|MySpace|Google|Exchange|SMS|Transcript)', raw_line, re.IGNORECASE)
                        or re.search(r'\d{1,2}/\d{1,2}/\d{2,4}$', raw_line)
                    ):
                        context_split = True
                    if current_block and context_split:
                        message_blocks.append(current_block)
                        current_block = []
                    last_dt = None
                    last_author = None
                    # Context clue: always start a new conversation block
                    # Split on context clue lines (section headers, lines containing 'Text Messaging', 'Email Exchange', or ending with a date)
                    context_split = False
                    if (
                        re.search(r'(Text Messaging|Email Exchange|Meetme.com|Chat Log|Conversation|Session|Log|Message|IMs?|AIM|Yahoo|MSN|ICQ|Skype|Gmail|Hotmail|Facebook|MySpace|Google|Exchange|SMS|Transcript)', raw_line, re.IGNORECASE)
                        or re.search(r'\d{1,2}/\d{1,2}/\d{2,4}$', raw_line)
                    ):
                        context_split = True
                    if current_block and context_split:
                        message_blocks.append(current_block)
                        current_block = []
                    last_dt = None
                    last_author = None

            if current_block:
                message_blocks.append(current_block)


            # Each message block is now a conversation candidate (split on context clues and 3-hour gaps)
            file_conversations = [block for block in message_blocks if block]



            # Robust normalization: ignore case, whitespace, punctuation for grouping
            def robust_normalize(username):
                if not username:
                    return None
                cleaned = username.strip().lower()
                cleaned = re.sub(r'\s*\([^)]*$', '', cleaned)
                cleaned = re.sub(r'\s*\([^)]*\)$', '', cleaned)
                cleaned = re.sub(r'\s*\([^)]*[\d:/apm -][^)]*\)?$', '', cleaned)
                cleaned = re.sub(r'\([^)]*\)', '', cleaned)
                cleaned = re.sub(r'\s+\d{1,2}:\d{2}(?::\d{2})?\s*(am|pm)?$', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'[^a-z0-9_.@&\-]+', '', cleaned)  # Remove all non-alphanum/punct
                return cleaned if cleaned else None

            def normalize_username(username):
                # Use previous logic for filtering, but robust_normalize for grouping
                if not username:
                    return None
                cleaned = username.strip()
                cleaned = re.sub(r'\s*\([^)]*$', '', cleaned)
                cleaned = re.sub(r'\s*\([^)]*\)$', '', cleaned)
                cleaned = re.sub(r'\s*\([^)]*[\d:/APMapm -][^)]*\)?$', '', cleaned)
                cleaned = re.sub(r'\([^)]*\)', '', cleaned)
                cleaned = re.sub(r'\s+\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM)?$', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.rstrip(':').strip()
                if not cleaned:
                    return None
                context_clues = [
                    r'^Email Exchange(\s|$)', r'^OFFLINE(\s|$)', r'^Text Messaging(\s|$)', r'^Meetme.com(\s|$)',
                    r'^Chat Log(\s|$)', r'^Conversation(\s|$)', r'^Session(\s|$)', r'^Log(\s|$)', r'^Message(\s|$)',
                    r'^IMs?(\s|$)', r'^AIM(\s|$)', r'^Yahoo(\s|$)', r'^MSN(\s|$)', r'^ICQ(\s|$)', r'^Skype(\s|$)',
                    r'^Gmail(\s|$)', r'^Hotmail(\s|$)', r'^Facebook(\s|$)', r'^MySpace(\s|$)', r'^Google(\s|$)',
                    r'^Exchange(\s|$)', r'^SMS(\s|$)', r'^Transcript(\s|$)', r'^\d{1,2}/\d{1,2}/\d{2,4}$',
                ]
                for clue in context_clues:
                    if re.match(clue, cleaned, re.IGNORECASE):
                        return None
                if re.match(r'.+?: .+', cleaned):
                    return None
                if re.match(r'^\d+$', cleaned):
                    return None
                if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', cleaned):
                    return None
                if len(cleaned) > 60:
                    return None
                if cleaned.count(' ') > 7:
                    return None
                if not re.match(r'^[A-Za-z0-9_.@&\- ]+$', cleaned):
                    return None
                return cleaned if cleaned else None


            # Collect all unique robust-normalized participants from all message blocks, filtering out message fragments
            all_participants = set()
            for block in message_blocks:
                for m in block:
                    norm = normalize_username(m.get('author'))
                    robust = robust_normalize(m.get('author'))
                    # Filter out message fragments and context lines
                    if norm and robust and not date_line_re.match(norm):
                        # Exclude if norm is a message fragment
                        if len(norm) > 30 and (" " in norm or norm.lower() in m.get('text', '').lower()):
                            continue
                        # Exclude if norm is a common phrase from the text
                        if norm.lower() in ["no nothing", "thinking of meeting u", "thts the reason i said", "well i know u r not baby", "well i try to meet someone personally"]:
                            continue
                        all_participants.add(norm)
            metadata['all_participants'] = sorted(all_participants)

            # Group by robust-normalized participant identities
            final_conversations = []
            file_conversations = [block for block in message_blocks if block]
            from collections import Counter
            for conv in file_conversations:
                # Use robust normalization for grouping
                robust_authors = [robust_normalize(m['author']) for m in conv if robust_normalize(m['author'])]
                if not robust_authors:
                    continue
                # Find the two most common robust-normalized authors
                author_counts = Counter(robust_authors)
                most_common = [a for a, _ in author_counts.most_common(2)]
                # Only keep if exactly two main participants and at least 6 messages from them
                filtered_msgs = [m for m in conv if robust_normalize(m['author']) in most_common]
                filtered_authors = [robust_normalize(m['author']) for m in filtered_msgs if robust_normalize(m['author'])]
                filtered_participants = sorted(set(filtered_authors))
                if len(filtered_participants) == 2 and len(filtered_authors) >= 6:
                    clean_messages = []
                    # Find the best contextual date for this block
                    # Use the most recent contextual_date or current_date seen in the block
                    block_contextual_date = None
                    block_current_date = None
                    # Search for contextual date cues in the block
                    for msg in filtered_msgs:
                        # Try to find a contextual date from metadata contextual_lines near this message
                        msg_ln = msg.get('line_number')
                        for ctx in metadata.get('contextual_lines', []):
                            if 'date' in ctx and ctx['date'] != 'N/A' and abs(ctx['line_number'] - msg_ln) < 50:
                                block_contextual_date = ctx['date']
                                break
                        if block_contextual_date:
                            break
                    # Fallback to file_date if no contextual date found
                    block_contextual_date = block_contextual_date or file_date or 'N/A'
                    for m in filtered_msgs:
                        norm_author = normalize_username(m['author'])
                        # If timestamp is N/A, try to assign contextual date
                        if m.get('datetime'):
                            try:
                                ts = m['datetime'].isoformat(sep=' ')
                            except Exception:
                                ts = str(m['datetime'])
                        else:
                            # Try to use contextual date and any time info in raw_datetime
                            raw_dt = m.get('raw_datetime')
                            time_part = None
                            if raw_dt and isinstance(raw_dt, str):
                                # Try to extract time from raw_datetime
                                import re as _re
                                time_match = _re.search(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)', raw_dt, _re.IGNORECASE)
                                if time_match:
                                    time_part = time_match.group(1)
                            if block_contextual_date != 'N/A' and time_part:
                                # Try to parse full datetime
                                from datetime import datetime as _dt
                                dt_str = f"{block_contextual_date} {time_part}"
                                dt_obj = parse_datetime(dt_str)
                                if dt_obj:
                                    ts = dt_obj.isoformat(sep=' ')
                                else:
                                    ts = f"{block_contextual_date} {time_part}"
                            elif block_contextual_date != 'N/A':
                                ts = block_contextual_date
                            else:
                                ts = 'N/A'
                        clean_messages.append({
                            'author': norm_author,
                            'text': m['text'],
                            'line_number': m['line_number'],
                            'timestamp': ts
                        })
                    final_conversations.append({
                        'participants': filtered_participants,
                        'messages': clean_messages
                    })
                else:
                    total_conv_skipped += 1

            # Save detections to file
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"PJ_{base_filename}.json"
            output_path = os.path.join(pj_output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': metadata,
                    'conversations': final_conversations
                }, f, indent=2)


            kept = len(final_conversations)
            if kept:
                print(f"  {filename}: {kept} conversations kept")
                total_kept += kept
            else:
                print(f"  {filename}: 0 conversations kept")
                total_skipped += 1

            # Debug output for selected file
            if DEBUG_MODE and filename == DEBUG_FILE:
                print("\n--- DEBUG STATS FOR", filename, "---")
                print("Participants:", metadata.get('all_participants', []))
                print("# Contextual lines:", len(metadata.get('contextual_lines', [])))
                for i, ctx in enumerate(metadata.get('contextual_lines', [])):
                    print(f"  Context {i+1}: line {ctx['line_number']} - {ctx['text'][:60]}")
                print("# Conversations:", len(final_conversations))
                for i, conv in enumerate(final_conversations):
                    print(f"  Conversation {i+1}: {len(conv['messages'])} messages, participants: {conv['participants']}")
                    if len(conv['messages']) > 0:
                        print(f"    First message: {conv['messages'][0]['text'][:60]}")
                        print(f"    Last message: {conv['messages'][-1]['text'][:60]}")
                print("-----------------------------\n")

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            total_skipped += 1

    print(f"  PJ Total Conversations - Kept: {total_kept}, Skipped: {total_conv_skipped} | Files with no conversations after filter: {total_skipped}")

if __name__ == "__main__":
    filter_conversations()