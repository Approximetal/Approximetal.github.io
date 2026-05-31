import assert from "node:assert/strict";
import test from "node:test";
import { formatDate } from "../src/utils/others.ts";

test("formatDate formats supported output variants", () => {
  assert.equal(formatDate("2024-02-03"), "Feb 3, 2024");
  assert.equal(formatDate("2024-02-03", 1), "Feb 3");
  assert.equal(formatDate("2024-02-03", 2), "2024/02");
});
