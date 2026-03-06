/**
 * AGE helper unit tests — agtype parsing.
 *
 * Note: The actual graph operations require a running Postgres with AGE.
 * These tests only cover the parsing logic.
 */

import { describe, it, expect } from 'vitest';

import { parseAgtype } from './age.js';

describe('parseAgtype', () => {
  it('parses a vertex', () => {
    const raw = '{"id": 844424930131969, "label": "Entity", "properties": {"name": "Tokyo", "type": "location"}}::vertex';
    const result = parseAgtype(raw);

    expect(result).toEqual({
      id: 844424930131969,
      label: 'Entity',
      properties: { name: 'Tokyo', type: 'location' },
    });
  });

  it('parses an edge', () => {
    const raw = '{"id": 1125899906842625, "label": "MENTIONS", "end_id": 844424930131969, "start_id": 844424930131970, "properties": {}}::edge';
    const result = parseAgtype(raw);

    expect(result).toEqual({
      id: 1125899906842625,
      label: 'MENTIONS',
      end_id: 844424930131969,
      start_id: 844424930131970,
      properties: {},
    });
  });

  it('parses a string scalar', () => {
    expect(parseAgtype('"hello"')).toBe('hello');
  });

  it('parses a number scalar', () => {
    expect(parseAgtype('42')).toBe(42);
  });

  it('parses a boolean', () => {
    expect(parseAgtype('true')).toBe(true);
    expect(parseAgtype('false')).toBe(false);
  });

  it('handles null', () => {
    expect(parseAgtype(null)).toBeNull();
    expect(parseAgtype(undefined)).toBeNull();
  });

  it('handles unparseable string', () => {
    expect(parseAgtype('just a string')).toBe('just a string');
  });
});
