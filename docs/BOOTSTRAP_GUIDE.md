# Bootstrap & YAML Validation Guide

## What Changed

### 1. URLValidationAgent Integration
- **Validates URLs BEFORE synthesis** - catches wrong docs early
- **Search refinement loop** - retries with better queries if validation fails
- **Reduces answer refinement cycles** - LinuxExpert gets good docs from start

### 2. Jira Integration (Default: OFF)
- **Default:** Jira updates are DISABLED
- **Enable with:** `--enable-jira` flag
- **Dry-run mode:** `--dry-run-integrations` shows preview without posting

### 3. YAML Validation Script
- **Validate existing YAMLs** without full re-extraction
- **Auto-fix option** to search for better URLs
- **Dry-run mode** to preview changes

---

## Re-Running Bootstrap with URL Validation

The `extract_jira_tickets.py` script now includes URLValidationAgent automatically.

### Full Re-Extraction (Recommended for BOOTLOADER patterns)

```bash
cd /home/emackey/Work/rhel-lightspeed/HEAL

# Re-extract specific pattern with URL validation
uv run python src/heal/bootstrap/extract_jira_tickets.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --force-reextract

# Re-extract all BOOTLOADER patterns
for pattern in BOOTLOADER_*; do
    uv run python src/heal/bootstrap/extract_jira_tickets.py \
        --tickets $(grep -l "$pattern" config/patterns/*.yaml | head -1 | xargs basename -s .yaml) \
        --force-reextract
done
```

### What Happens During Extraction

```
1. LinuxExpert forms hypothesis
2. SolrExpert searches for verification docs
3. ✨ URLValidationAgent validates retrieved URLs
   - If valid: Continue to synthesis
   - If invalid: Retry search with suggested queries (max 2 attempts)
4. LinuxExpert synthesizes answer from VALIDATED docs
5. AnswerReviewAgent checks answer quality
```

---

## Validating Existing Pattern YAMLs (No Re-Extraction)

Use `scripts/validate_yaml_urls.py` to **update pattern YAMLs in-place** without full re-extraction.

### How It Works

1. **Reads pattern YAML** from `config/patterns/{PATTERN_ID}.yaml`
2. **For each ticket**: Searches Solr with the query to get REAL docs
3. **Validates** those docs match the query
4. **Updates expected_urls** in the pattern YAML if better URLs found
5. **Saves** changes back to the same pattern YAML file

### Read-Only Validation (Just Report Issues)

```bash
# Validate all patterns (no changes)
uv run python scripts/validate_yaml_urls.py

# Validate specific pattern
uv run python scripts/validate_yaml_urls.py --pattern BOOTLOADER_GRUB_ISSUES
```

**Output:**
```
[RSPEED-1723] Searching Solr and validating URLs...
  Query: How to update GRUB in RHEL 9?
  Current URLs: 3
  Found 5 docs from Solr
  Validation score: 0.88
  Validation passes: True
  ✅ Validation passed, URLs differ from current
  📝 Updated RSPEED-1723 URLs in memory

Validation Summary: BOOTLOADER_GRUB_ISSUES
Total tickets: 3
  ✅ Passed (unchanged): 1
  📝 Updated: 2
  ❌ Failed: 0

URL Changes:
  RSPEED-1723: updated
    Score: 0.88
    Old URLs (3): ['solutions/3486741', 'solutions/12345']...
    New URLs (3): ['solutions/1521', 'solutions/1212383']...
```

### Auto-Fix Mode (Retry with Better Queries)

```bash
# Dry-run (preview changes)
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix \
    --dry-run

# Apply fixes (creates .yaml.bak backup)
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix
```

**What auto-fix does:**
1. Searches Solr with query
2. Validates retrieved docs
3. If validation fails → retries with URLValidationAgent's suggested queries
4. Updates pattern YAML with best URLs found
5. Creates backup (.yaml.bak)

---

## Testing Jira Integration (Dry-Run Mode)

The fix loop now has Jira integration with dry-run mode to preview comments.

### Preview Jira Comments (No Posting)

```bash
# Run fix loop with Jira preview (default: Jira disabled)
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --dry-run-integrations

# Or explicitly enable Jira in dry-run mode
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --enable-jira --dry-run-integrations
```

**Output:**
```
================================================================================
JIRA INTEGRATION: Updating Tickets
================================================================================
   Pattern: BOOTLOADER_GRUB_ISSUES
   Tickets: 3
   Dry Run: True

[DRY RUN] Would post to RSPEED-1723
  Preview: ## 🤖 Automated Pattern Fix: Bootloader Grub Issues...
  Full comment saved to: .diagnostics/BOOTLOADER_GRUB_ISSUES/jira_preview_RSPEED-1723.md

================================================================================
[DRY RUN] Preview: 3 comments
Full preview saved to: .diagnostics/BOOTLOADER_GRUB_ISSUES/JIRA_COMMENTS_PREVIEW.md
To post these comments, re-run with: --enable-jira
================================================================================
```

### Review Preview File

```bash
cat .diagnostics/BOOTLOADER_GRUB_ISSUES/JIRA_COMMENTS_PREVIEW.md
```

Shows full formatted comments that would be posted to each ticket.

### Actually Post to Jira (When Ready)

```bash
# Enable Jira updates (NO dry-run)
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --enable-jira
```

---

## Recommended Workflow for BOOTLOADER Patterns

### Option 1: Quick YAML Fix (RECOMMENDED - No Re-Extraction)

**Best for:** Just fixing URLs in existing pattern YAMLs

```bash
# Step 1: Validate + auto-fix pattern YAML in-place (dry-run first)
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix \
    --dry-run

# Step 2: Review proposed changes, then apply
uv run python scripts/validate_yaml_urls.py \
    --pattern BOOTLOADER_GRUB_ISSUES \
    --auto-fix

# Step 3: Review updated YAML
git diff config/patterns/BOOTLOADER_GRUB_ISSUES.yaml

# Step 4: Run fix loop with Jira preview
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --dry-run-integrations

# Step 5: If good, run for real
./runners/fix.sh BOOTLOADER_GRUB_ISSUES --enable-jira
```

**What this does:**
- ✅ Updates `expected_urls` in pattern YAML directly
- ✅ Uses URLValidationAgent to verify URLs match queries
- ✅ Creates backup (.yaml.bak)
- ✅ Fast - no full re-extraction needed
- ✅ Preserves existing `expected_response` and other fields

### Option 2: Full Re-Extraction (If you need new expected_response too)

**Best for:** Regenerating expected answers, not just URLs

```bash
# Step 1: Re-extract tickets (goes to extracted_tickets.yaml)
uv run python src/heal/bootstrap/extract_jira_tickets.py \
    --tickets RSPEED-1723,RSPEED-1724,RSPEED-1725 \
    --force-reextract

# Step 2: Run pattern discovery to create pattern YAMLs
# (You'll need to run pattern discovery script to split extracted_tickets.yaml)

# Step 3: Run fix loop
./runners/fix.sh BOOTLOADER_GRUB_ISSUES
```

**Note:** This is overkill if you just want to fix URLs. Use Option 1 instead.

---

## Integration Flags Reference

### fix.sh Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-jira` | OFF | Enable Jira comment updates |
| `--create-pr` | OFF | Create GitHub PR after fix |
| `--dry-run-integrations` | OFF | Preview Jira/PR without executing |
| `--include-judge-reasoning` | OFF | Include LLM judge critique in diagnostics |

### Examples

```bash
# Default: No integrations
./runners/fix.sh PATTERN_ID

# Preview Jira comments (no posting)
./runners/fix.sh PATTERN_ID --dry-run-integrations

# Enable Jira updates
./runners/fix.sh PATTERN_ID --enable-jira

# Enable both Jira and PR
./runners/fix.sh PATTERN_ID --enable-jira --create-pr

# A/B test with judge reasoning
./runners/fix.sh PATTERN_ID --include-judge-reasoning
```

---

## Output Files

### Jira Integration

| File | When Created | Purpose |
|------|--------------|---------|
| `.diagnostics/{pattern}/JIRA_COMMENTS_PREVIEW.md` | Dry-run mode | Preview of comments |
| `.diagnostics/{pattern}/JIRA_COMMENTS_FALLBACK.md` | Post failures | Failed comments for manual copy |

### YAML Validation

| File | When Created | Purpose |
|------|--------------|---------|
| `config/patterns/{pattern}.yaml.bak` | Auto-fix applied | Backup before update |

---

## Troubleshooting

### "No documents retrieved" during validation

**Cause:** URLs in YAML are just strings, not full documents

**Solution:** Use auto-fix mode to search Solr for actual docs:
```bash
uv run python scripts/validate_yaml_urls.py --pattern X --auto-fix
```

### "Jira integration failed"

**Check:**
1. MCP Atlassian server is running
2. Claude Agent SDK is installed: `uv pip list | grep claude-agent-sdk`
3. Try dry-run first to see what would be posted

### "URL validation keeps failing"

**Check:**
1. Are expected_urls in YAML actually relevant to the query?
2. Review validation issues in output
3. Try full re-extraction instead of just validation:
   ```bash
   uv run python src/heal/bootstrap/extract_jira_tickets.py --pattern X --force-reextract
   ```

---

## Next Steps

1. **Re-extract BOOTLOADER patterns** with URL validation
2. **Preview Jira integration** in dry-run mode
3. **Validate other patterns** using validation script
4. **A/B test judge reasoning** flag to measure impact
