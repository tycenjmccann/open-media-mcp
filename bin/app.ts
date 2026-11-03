#!/usr/bin/env node
import * as fs from "fs";
import * as path from "path";
import { App } from "aws-cdk-lib";
import { OpenMediaMcpStack } from "../lib/open-media-mcp-stack";

const app = new App();
// cdk.local.json (gitignored): your deployment's media:* settings. -c flags still win.
const local = path.join(__dirname, "..", "cdk.local.json");
if (fs.existsSync(local)) {
  for (const [k, v] of Object.entries(JSON.parse(fs.readFileSync(local, "utf8")))) {
    if (app.node.tryGetContext(k) === undefined) app.node.setContext(k, v);
  }
}
new OpenMediaMcpStack(app, app.node.tryGetContext("media:stackName") ?? "OpenMediaMcp", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
  description: "open-media-mcp: open-weight video and audio generation as MCP tools on Amazon Bedrock AgentCore Gateway",
});
