#!/usr/bin/env node
'use strict';

/* Collect chosen battles from several recordings into one small JSONL fixture. Run:
 * node reference/merge-recordings.cjs --out tests/fixtures/diff-regress.jsonl.gz \
 *     --tail 12 work/diff/pool-seed2.jsonl.gz:66@4 work/diff/seed6-boosts.jsonl.gz:54@38 ...
 *
 * Each source is PATH:INDEX or PATH:INDEX@STEP. With @STEP the battle keeps its steps up to
 * STEP + --tail (the first mismatching step plus some steps after it), which keeps the file
 * small; without it the whole battle is kept. Battles are renumbered 0, 1, 2, ... in the order
 * given, and each records its origin in "source". The header comes from the first source with
 * the new count; every source must use the same Showdown commit and format.
 */

const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');
const { gunzipSync, gzipSync } = require('node:zlib');

function readRecording(file) {
	const raw = fs.readFileSync(file);
	const text = (file.endsWith('.gz') ? gunzipSync(raw) : raw).toString('utf8');
	const [header, ...battles] = text.split('\n').filter(line => line.trim()).map(JSON.parse);
	return { header, battles };
}

function parseSource(spec) {
	const match = /^(.*):(\d+)(?:@(\d+))?$/.exec(spec);
	if (!match) throw new Error(`Expected PATH:INDEX or PATH:INDEX@STEP, got ${spec}`);
	return { file: match[1], index: Number(match[2]), step: match[3] === undefined ? null : Number(match[3]) };
}

function merge(sources, tail = 12) {
	const cache = new Map();
	let header = null;
	const battles = [];
	for (const spec of sources.map(parseSource)) {
		if (!cache.has(spec.file)) cache.set(spec.file, readRecording(spec.file));
		const recording = cache.get(spec.file);
		const battle = recording.battles.find(candidate => candidate.index === spec.index);
		if (!battle) throw new Error(`${spec.file} has no battle ${spec.index}`);
		if (!header) header = recording.header;
		if (recording.header.showdownCommit !== header.showdownCommit || recording.header.format !== header.format) {
			throw new Error(`${spec.file} uses a different Showdown commit or format`);
		}
		const copy = structuredClone(battle);
		if (spec.step !== null) copy.steps = copy.steps.slice(0, spec.step + tail + 1);
		copy.source = { file: path.basename(spec.file), index: spec.index, ...(spec.step === null ? {} : { step: spec.step }) };
		copy.index = battles.length;
		battles.push(copy);
	}
	if (!header) throw new Error('No sources given');
	return [{ ...header, count: battles.length, merged: true }, ...battles];
}

function main() {
	const { values, positionals } = parseArgs({
		options: { out: { type: 'string' }, tail: { type: 'string', default: '12' } },
		allowPositionals: true,
	});
	if (!values.out) throw new Error('--out is required');
	const records = merge(positionals, Number(values.tail));
	const text = records.map(record => JSON.stringify(record)).join('\n') + '\n';
	fs.mkdirSync(path.dirname(path.resolve(values.out)), { recursive: true });
	fs.writeFileSync(values.out, values.out.endsWith('.gz') ? gzipSync(text, { level: 9 }) : text);
	console.log(`Wrote ${values.out}: ${records.length - 1} battles, ${text.length} bytes before compression`);
}

module.exports = { merge, parseSource, readRecording };
if (require.main === module) main();
