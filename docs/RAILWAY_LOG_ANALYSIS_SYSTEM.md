# Railway Log Analysis System

## Overview

The Railway Log Analysis System is an automated solution that feeds Railway logs into a Cursor background agent daily to analyze errors, detect patterns, and automatically fix common issues.

## Architecture

```
Railway Service → Log Fetcher → Log Analyzer → Auto Fixer → GitHub PR
                                      ↓
                               Database Storage ← Monitoring Dashboard
```

### Components

1. **Railway Log Fetcher** (`src/monitoring/railway_log_fetcher.py`)
   - Fetches logs from Railway using CLI
   - Supports time-based filtering and multiple environments
   - Saves logs locally for analysis

2. **Log Analyzer** (`src/monitoring/log_analyzer.py`) 
   - Parses structured JSON and plain text logs
   - Detects error patterns using regex matching
   - Generates analysis reports with recommendations

3. **Auto Fixer** (`src/monitoring/auto_fixer.py`)
   - Automatically generates fixes for common error patterns
   - Creates pull requests for review
   - Handles safety checks and validation

4. **Daily Orchestrator** (`scripts/railway_log_analyzer.py`)
   - Coordinates the complete analysis pipeline
   - Manages the daily execution workflow
   - Integrates with database for historical tracking

## Features

### Automated Error Detection

The system detects and categorizes these error patterns:

- **API Rate Limiting**: Detects 429 errors and rate limit messages
- **Database Connection Issues**: Identifies connection timeouts and failures  
- **Authentication Errors**: Catches 401/403 and auth failures
- **Memory Issues**: Detects OOM kills and memory warnings
- **Timeout Errors**: Identifies connection and read timeouts
- **JSON Parsing Errors**: Catches malformed JSON responses
- **Missing Configuration**: Detects missing environment variables

### Automated Fixes

For safe, well-understood issues, the system can automatically:

- Add rate limiting with exponential backoff
- Implement timeout configuration constants
- Create robust JSON parsing utilities
- Generate error handling improvements

### Safety Features

- **Dry Run Mode**: Analyze without making changes
- **Manual Review Requirements**: Complex issues flagged for human review
- **Pull Request Workflow**: All fixes go through PR review process
- **Database Tracking**: Historical analysis data for trend monitoring

## Usage

### CLI Commands

```bash
# * Fetch logs from Railway
atb logs fetch --environment production --hours 24 --save

# * Analyze logs for errors
atb logs analyze --environment production --hours 24 --output report.md

# * Generate fixes for detected errors  
atb logs fix --report-file analysis_report.json --dry-run

# * Run complete daily pipeline
atb logs daily --environment production --hours 24
```

### Background Agent Setup

The system includes a Cursor background agent configuration (`environment.json`):

```json
{
  "install": "pip install -r requirements.txt && [install Railway CLI & GitHub CLI]",
  "terminals": [
    {
      "name": "Daily Log Analyzer", 
      "command": "python scripts/railway_log_analyzer.py --environment production --hours 24"
    }
  ]
}
```

### GitHub Actions Workflow

Daily automated execution via GitHub Actions (`.github/workflows/daily-railway-log-analysis.yml`):

- **Schedule**: 06:00 UTC daily (after trading day)
- **Manual Trigger**: Supports manual execution with parameters
- **Environments**: Production, staging, development
- **Outputs**: Analysis reports, fix PRs, manual review issues

## Configuration

### Environment Variables

Required for Railway access:
```bash
RAILWAY_PROJECT_ID=your-project-id
RAILWAY_SERVICE_ID=your-service-id  # optional
GITHUB_TOKEN=your-github-token      # for PR creation
```

### Database Integration

The system integrates with the existing PostgreSQL database:

- Stores analysis results in `system_events` table
- Tracks error patterns and frequencies over time
- Enables trend analysis through monitoring dashboard

## Reports and Outputs

### Log Analysis Report

Generated markdown reports include:

- **Summary Statistics**: Total entries, error counts, error rates
- **Error Patterns**: Detailed breakdown with frequencies and severity
- **Performance Issues**: Memory, latency, and restart patterns  
- **Recommendations**: Actionable steps for issue resolution

### Pull Requests

Automated PRs include:

- **Descriptive Titles**: Clear indication of fixes applied
- **Detailed Descriptions**: Analysis summary and fix explanations
- **File Changes**: Code additions and configuration updates
- **Safety Notes**: Review guidelines and testing recommendations

### Manual Review Issues

For complex issues requiring human intervention:

- **Issue Creation**: Automatic GitHub issue creation
- **Categorization**: Clear labeling and priority assignment
- **Context**: Full error details and suggested approaches
- **Tracking**: Links to original analysis and related PRs

## Integration with Existing Systems

### Monitoring Dashboard

The log analysis integrates with the existing monitoring dashboard:

- Historical error trend visualization
- Real-time error rate monitoring  
- Analysis result display
- Performance impact tracking

### Database Logging

Leverages existing database infrastructure:

- Uses `SystemEvent` model for analysis storage
- Integrates with `DatabaseManager` for consistent access
- Maintains audit trail of all analysis runs

### Configuration System

Follows project configuration patterns:

- Uses existing config providers
- Respects Railway environment detection
- Integrates with secrets management

## Security Considerations

### Sensitive Data Handling

- **Log Redaction**: Sensitive data is redacted from logs before analysis
- **Secure Storage**: Temporary log files are cleaned up after processing
- **Access Control**: Requires proper Railway and GitHub authentication

### Code Safety

- **Review Process**: All fixes go through PR review before merge
- **Validation**: Automated syntax and safety checks
- **Rollback**: Git-based rollback capability for any issues

## Monitoring and Alerting

### Success Metrics

- Daily analysis completion rate
- Error pattern detection accuracy
- Fix application success rate
- PR merge rate and review time

### Failure Handling

- **Graceful Degradation**: Continues analysis even if some steps fail
- **Error Logging**: Comprehensive error tracking and reporting
- **Notification**: GitHub Actions notifications for failures
- **Retry Logic**: Built-in retry for transient failures

## Deployment

### Quick Setup

Run the automated setup script:

```bash
python scripts/setup_railway_log_analysis.py
```

This script will:
- Verify Railway CLI installation and authentication
- Check GitHub CLI setup
- Validate environment variables
- Create required directories
- Test Railway connection and log access
- Validate log analysis functionality

### Prerequisites

1. **Railway CLI**: Installed and authenticated
2. **GitHub CLI**: Installed and authenticated  
3. **Python Dependencies**: All requirements installed
4. **Database Access**: PostgreSQL connection configured

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# * Required
RAILWAY_PROJECT_ID=your-railway-project-id

# * Optional but recommended
RAILWAY_SERVICE_ID=your-railway-service-id
GITHUB_TOKEN=your-github-token
```

### Manual Testing

```bash
# * Test log fetching
atb logs fetch --environment staging --hours 1 --save

# * Test analysis
atb logs analyze --environment staging --hours 1 --output test_report.md

# * Test full pipeline (dry run)
atb logs daily --environment staging --hours 1 --dry-run
```

### Automated Deployment

The GitHub Actions workflow handles automated deployment:

1. **Daily Execution**: Runs automatically at 06:00 UTC
2. **Environment Setup**: Installs all required tools
3. **Authentication**: Uses repository secrets for access
4. **Result Storage**: Saves artifacts for review and debugging

## Troubleshooting

### Common Issues

#### Railway CLI Authentication
```bash
# * Check authentication
railway whoami

# * Re-authenticate if needed
railway login
```

#### Missing Logs
```bash
# * Verify service is running
railway status

# * Check recent deployments
railway deployment list

# * Test log access manually
railway logs --since 1h
```

#### GitHub PR Creation Failures
```bash
# * Check GitHub CLI authentication
gh auth status

# * Verify repository permissions
gh repo view --json permissions
```

### Debug Commands

```bash
# * Test individual components
python src/monitoring/railway_log_fetcher.py --hours 1
python src/monitoring/log_analyzer.py --log-file logs/test.log
python src/monitoring/auto_fixer.py --report-file report.json --dry-run

# * Check database connectivity
atb db verify

# * View recent analysis results
atb logs analyze --log-file logs/railway/production/railway_logs_latest.log
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**: ML-based error pattern detection
2. **Advanced Fix Generation**: More sophisticated code modification
3. **Performance Optimization**: Automated performance tuning suggestions
4. **Integration Testing**: Automated testing of generated fixes
5. **Trend Analysis**: Long-term error trend analysis and prediction

### Configuration Improvements

1. **Custom Pattern Definitions**: User-defined error patterns
2. **Fix Templates**: Customizable fix generation templates
3. **Notification Channels**: Slack, Discord, email integration
4. **Scheduling Options**: Flexible analysis scheduling

## Contributing

### Adding New Error Patterns

1. Define pattern in `RailwayLogAnalyzer.known_error_patterns`
2. Implement fix in `AutoFixer.fix_implementations` 
3. Add tests for pattern detection and fix generation
4. Update documentation with new pattern details

### Testing

```bash
# * Unit tests for log analysis
python tests/run_tests.py unit -k test_log_analyzer

# * Integration tests with Railway
python tests/run_tests.py integration -k test_railway_logs

# * End-to-end pipeline testing
python scripts/railway_log_analyzer.py --dry-run --hours 1
```

---

*This system addresses GitHub issue #222 by providing comprehensive automated Railway log analysis and error fixing capabilities.*