import fs from 'fs';
import path from 'path';
import moment from 'moment';
import simpleGit from 'simple-git';
import jsonfile from 'jsonfile';

const DATA_PATH = path.join(process.cwd(), 'data.json');
const git = simpleGit();

const commitMessages = [
  "Fix bug",
  "Add feature",
  "Refactor code",
  "Update dependencies",
  "Improve performance",
  "Add tests",
  "Documentation update",
  "Code cleanup",
  "Optimize algorithm",
];

// Function to generate a GitHub-compatible timestamp
function generateTimestamp(weeksBack = 0, daysBack = 0) {
  return moment().subtract(weeksBack, 'weeks').subtract(daysBack, 'days').format('YYYY-MM-DDTHH:mm:ssZ');
}

// Function to add a timestamp to data.json
async function addCommitDate(timestamp) {
  const data = await jsonfile.readFile(DATA_PATH);
  data.push({ date: timestamp });
  await jsonfile.writeFile(DATA_PATH, data, { spaces: 2 });
}

// Function to mark a commit at specific X (weeks) and Y (days) coordinates on the graph
async function markCommit(x, y) {
  const timestamp = generateTimestamp(x, y);
  await addCommitDate(timestamp);
  const message = commitMessages[Math.floor(Math.random() * commitMessages.length)];
  await git.commit(message, { '--date': timestamp, '--allow-empty': true });
  await git.push();
  console.log(`Committed to ${x} weeks, ${y} days back.`);
}

// Example: Run this to test a single commit (e.g., 2 weeks and 3 days back)
{/*
for (let i = 0; i < 5; i++) {
  await markCommit(2+i, i);
}
await markCommit(3, 3);
await markCommit(3, 4);
*/}

// For random "legit" commits (generates commits across the year, 2-6 per random day)
async function generateRandomCommits(numCommits = 100) {
  let data = await jsonfile.readFile(DATA_PATH);
  let commitsAdded = 0;
  const changes = []; // Track pending changes

  while (commitsAdded < numCommits) {
    const x = Math.floor(Math.random() * 53); // 53 weeks in a year
    const y = Math.floor(Math.random() * 7);  // 7 days in a week
    const base = moment().subtract(x, 'weeks').subtract(y, 'days');
    const commitsThisDay = Math.floor(Math.random() * 5) + 2; // 2-6 commits per day

    for (let j = 0; j < commitsThisDay && commitsAdded < numCommits; j++) {
      const randomHours = Math.floor(Math.random() * 24);
      const randomMinutes = Math.floor(Math.random() * 60);
      const timestamp = base.clone().add(randomHours, 'hours').add(randomMinutes, 'minutes').format('YYYY-MM-DDTHH:mm:ssZ');

      // Avoid duplicates
      if (data.some(d => d.date === timestamp)) continue;

      // Add a tiny "real" change: append to data.json (e.g., a log entry)
      const entry = { date: timestamp, note: `Work log: ${commitMessages[Math.floor(Math.random() * commitMessages.length)]}` };
      data.push(entry);
      changes.push({ timestamp, message: entry.note }); // For commit messages
      commitsAdded++;
    }
  }

  // Write all changes at once
  await jsonfile.writeFile(DATA_PATH, data, { spaces: 2 });

  // Stage once
  await git.add(DATA_PATH);

  // Batch-commit with backdates (simple-git loops over changes)
  for (const { timestamp, message } of changes) {
    const env = {
      GIT_AUTHOR_DATE: timestamp,
      GIT_COMMITTER_DATE: timestamp
    };
    await git.env(env).commit(message, [], { '--allow-empty': false }); // No empty now
  }

  // One big push
  await git.push({ '--force': true }); // Safe for solo private repo; rewrites history
  console.log(`${numCommits} random commits added and pushed!`);
}

// Uncomment to run random commits
generateRandomCommits(22);

