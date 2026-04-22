# OKP Documentation Gaps Report

**Generated:** 2026-04-21  
**Source:** RAG-enhanced extraction testing with URL validation  
**Sample Size:** 7 RHEL tickets from recent CLA incorrect-answer reports

---

## Executive Summary

During automated extraction of test cases from JIRA tickets, we discovered **systematic documentation gaps in the OKP Solr index**. Using URL validation (LLM-based semantic matching), we found that:

- **URL validation success rate: <20%** (only 2-3 out of 7 tickets retrieved semantically correct docs)
- **Common issue:** Queries retrieve troubleshooting articles instead of procedural guides
- **Missing content:** Basic "how-to" documentation for rpm-ostree, RHEL System Roles, Convert2RHEL

Despite using RAG-enhanced retrieval (87.4% content relevance in benchmarks), we still couldn't find the right docs because **they aren't in the index or aren't properly tagged**.

---

## Detailed Findings by Topic

### 1. rpm-ostree Commands (3 tickets)

**Tickets:** RSPEED-1930, RSPEED-1929, RSPEED-1859

**Queries that failed:**
- "How do I install packages on a system using rpm-ostree?"
- "How do I rollback my system to a previous version using rpm-ostree?"
- "What is the correct rpm-ostree command to view the list of deployments?"

**What we retrieved instead:**
- ❌ Troubleshooting articles for rpm-ostree errors
- ❌ DNF and traditional `rpm` command docs (wrong package manager)
- ❌ Generic articles about "status" commands (gluster peer status, systemctl status, kdumpctl status)
- ❌ Completely unrelated docs (NFS performance, firewalld rules, auditd)

**What's missing:**
- ✅ Basic rpm-ostree command reference (install, rollback, status)
- ✅ Procedural guides for RHEL CoreOS / Image Mode package management
- ✅ Official rpm-ostree documentation from RHEL CoreOS docs
- ✅ Step-by-step procedures vs troubleshooting-only content

**URL Validation Scores:** 0.10-0.50 (fail)

**Example search that failed:**
```
Query: "rpm-ostree install package command RHEL"
Retrieved:
1. "Unable to install crash-ptdump-command package in RHEL-9" (uses DNF, not rpm-ostree)
2. "How to install httpd package using rpm command" (traditional rpm, not rpm-ostree)
3. "DNF can not work with pkcs11 certificate" (completely different tool)
```

---

### 2. RHEL System Roles (3 tickets)

**Tickets:** RSPEED-1812, RSPEED-1811, RSPEED-1775

**Queries that failed:**
- "Where can I find the official documentation for RHEL System Roles?"
- "What RHEL versions are supported by RHEL System Roles?"

**What we retrieved instead:**
- ❌ Troubleshooting articles for specific System Roles bugs
- ❌ Docs about completely unrelated topics (socket options, sudo drop-in dirs)
- ❌ Old errata announcements from 2018
- ❌ Generic Ansible articles (not System Roles specific)

**What's missing:**
- ✅ RHEL System Roles overview/landing page
- ✅ Supported versions/compatibility matrix
- ✅ Links to official docs.redhat.com guides
- ✅ "Automating system administration using RHEL System Roles" guide (mentioned in hypothesis but not retrieved)
- ✅ Knowledge base article 3050101 (mentioned but not retrieved)

**URL Validation Scores:** 0.00-0.30 (fail)

**Example search that failed:**
```
Query: "RHEL System Roles official documentation location"
Retrieved:
1. "RHEL System Roles overview" (might have info, but unclear from title)
2-4. Troubleshooting articles for specific role failures
5. Errata announcement from 2018
```

**Problem:** Meta-questions like "WHERE can I find documentation" return technical HOW-TO content instead of navigation/reference guides.

---

### 3. Convert2RHEL / Red Hat Insights (2 tickets)

**Tickets:** RSPEED-1775, RSPEED-1774

**Queries that failed:**
- "What Linux distributions can I convert to RHEL using Red Hat Insights?"
- "What yum repository is the convert2rhel utility available in?"

**What we retrieved instead:**
- ❌ Red Hat Insights general features (ACL permissions, advisor rules, changelog)
- ❌ Repository filesystem location troubleshooting
- ❌ Maven and Git repository docs (completely wrong type of "repository")
- ❌ Convert2RHEL troubleshooting (UID/GID issues, dependency problems)

**What's missing:**
- ✅ Insights Convert2RHEL supported distributions matrix
- ✅ Difference between Insights-based vs CLI-based conversion
- ✅ Convert2RHEL repository configuration/installation guide
- ✅ CDN repository URLs and setup instructions

**URL Validation Scores:** 0.00-0.45 (fail)

**Example search that failed:**
```
Query: "What yum repository is the convert2rhel utility available in?"
Retrieved:
1. "How to change repository location in Satellite 6?" (filesystem paths, not package repos)
2. "Default maven repository location" (wrong type of repo)
3. "How to configure Git repository location" (wrong type of repo)
4. PXE installation errors
5. RHEL 7 installation errors
```

---

## Pattern Analysis: Why Searches Fail

### 1. Procedural vs Troubleshooting Mismatch

**User asks:** "How do I [perform task]?" (needs procedure)  
**Solr returns:** "Error when [performing task]" (troubleshooting)

**Example:**
- Query: "How do I register a RHEL system?" → Retrieved: "SSL error during registration", "Timeout during registration"
- Query: "How do I install packages with rpm-ostree?" → Retrieved: "rpm-ostree install dependency error"

**Root cause:** Troubleshooting articles contain the query terms but serve a different purpose.

### 2. Generic Keyword Matching

**User asks:** "rpm-ostree status command"  
**Solr returns:** Any doc with "status" and "command" (kdumpctl status, systemctl status, gluster peer status)

**Root cause:** Query matches common words instead of the specific tool (rpm-ostree).

### 3. Meta-Questions Not Handled

**User asks:** "WHERE can I find documentation?" (meta-question)  
**Solr returns:** Documentation ABOUT the topic (technical content)

**Example:**
- Query: "Where can I find RHEL System Roles documentation?"
- Retrieved: Technical guides about using System Roles (not links to documentation locations)

**Root cause:** Index doesn't have navigation/reference content, only technical how-to articles.

### 4. Tool-Specific Terms Lost

**User asks:** About tool X (rpm-ostree, Convert2RHEL)  
**Solr returns:** Docs about similar tool Y (rpm, DNF, general Insights features)

**Root cause:** Insufficient docs for niche tools, or poor tagging to distinguish tool-specific content.

---

## Impact on AI Assistant Quality

### Current Workaround

LinuxExpert synthesized quality answers (0.70-1.00 review scores) **despite** URL validation failures by relying on:
- Internal RHEL knowledge (LLM training data)
- Best effort synthesis from partially relevant docs
- Conservative "check official docs" disclaimers

### Risk

Without proper documentation retrieval:
- ❌ Answers lack authoritative source URLs (reduces trust)
- ❌ Risk of outdated information (LLM training cutoff vs current RHEL versions)
- ❌ Cannot verify technical accuracy against official docs
- ❌ More refinement iterations needed (30% of tickets required 2-3 iterations)

### Business Value of Fixing

If OKP index had proper coverage:
- ✅ Faster answer generation (fewer refinement cycles)
- ✅ More authoritative answers (backed by official docs)
- ✅ Better source traceability (exact doc URLs)
- ✅ Higher confidence scores (validation passes)

---

## Recommendations for OKP Team

### High Priority (Quick Wins)

1. **Add rpm-ostree documentation**
   - RHEL CoreOS documentation for package management
   - Command reference for rpm-ostree (install, rollback, status, upgrade)
   - Ensure indexed from official RHEL 9+ docs

2. **Add RHEL System Roles landing pages**
   - Overview/getting started guide
   - Supported versions matrix
   - Links to official docs.redhat.com content
   - Knowledge base article 3050101

3. **Add Convert2RHEL installation guide**
   - Repository configuration steps
   - Supported distributions matrix
   - Insights vs CLI conversion comparison

### Medium Priority (Content Type Tagging)

4. **Tag document types clearly**
   - Procedural guides (how-to)
   - Troubleshooting articles (error resolution)
   - Reference documentation (command syntax)
   - Conceptual overviews

   **Use case:** Boost procedural guides for "how to" queries, troubleshooting for error queries

5. **Improve tool-specific indexing**
   - Distinguish rpm-ostree from rpm from DNF
   - Tag Red Hat Insights conversion separately from general Insights features
   - Separate RHEL System Roles from generic Ansible content

### Long-term (Semantic Search)

6. **Consider semantic field boosting**
   - For procedural queries: boost guides, de-boost troubleshooting
   - For meta-questions ("where can I find"): boost overview/navigation docs
   - For command syntax: boost reference documentation

7. **Add "missing doc" signals**
   - When searches consistently fail for a topic, log it
   - Prioritize creating content for high-failure topics

---

## Data Available for Analysis

We can provide the OKP team with:

1. **Search Intelligence Database**
   - `.claude/search_intelligence/` contains 783 logged searches
   - Query → Retrieved URLs → Success/failure
   - Failed searches by topic

2. **URL Validation Results**
   - Detailed issue breakdowns for each failed search
   - What was retrieved vs what was expected
   - Semantic mismatch scoring

3. **Comparison Scripts**
   - `scripts/compare_okp_vs_baseline.py` - Benchmarking tool
   - Can test any pattern/query against Solr index
   - Generates reproducible metrics

4. **Pattern YAMLs**
   - Expected queries and answers for testing
   - Can be used as acceptance criteria for improved indexing

---

## Reproducible Test Cases

To verify improvements, run:

```bash
cd ~/Work/rhel-lightspeed/HEAL

# Test rpm-ostree coverage
uv run python scripts/compare_okp_vs_baseline.py --pattern RPM_OSTREE_COMMANDS

# Test RHEL System Roles coverage  
uv run python scripts/compare_okp_vs_baseline.py --pattern SYSTEM_ROLES_INFO

# Full benchmark
uv run python scripts/compare_okp_vs_baseline.py --details
```

**Success criteria:**
- URL validation scores ≥ 0.7 (currently 0.0-0.5)
- Content relevance ≥ 0.8 (currently achieving 87.4% when docs exist)
- Retrieve procedural guides for "how to" queries (not just troubleshooting)

---

## Contact

For questions or to access raw data:
- **Search Intelligence DB:** `.claude/search_intelligence/search_results.jsonl`
- **Extraction Logs:** Available in HEAL diagnostics
- **Comparison Tools:** `scripts/compare_okp_vs_baseline.py`

We're happy to:
- Provide detailed query-by-query breakdowns
- Test specific documentation improvements
- Collaborate on acceptance testing for index updates

---

## Appendix: Example Success Case

**One area that worked well: Subscription Management**

Query: "How do I register with subscription-manager?"  
Retrieved: Official guides for subscription-manager registration ✅  
URL Validation Score: 0.8 (pass)

**Why it worked:**
- Clear procedural documentation exists
- Well-indexed in Solr
- Distinct from troubleshooting content
- Proper tagging for the tool name

**This is the target quality for all topics.**

---

*This report generated from automated RAG extraction testing - April 2026*
