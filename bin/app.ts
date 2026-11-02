#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { OpenMediaMcpStack } from "../lib/open-media-mcp-stack";

const app = new App();
new OpenMediaMcpStack(app, app.node.tryGetContext("media:stackName") ?? "OpenMediaMcp", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
  description: "open-media-mcp: open-weight video generation as MCP tools on Amazon Bedrock AgentCore Gateway",
});
