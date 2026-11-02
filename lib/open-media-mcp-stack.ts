import * as fs from "fs";
import * as path from "path";
import {
  Annotations,
  Aws,
  CfnOutput,
  Duration,
  RemovalPolicy,
  Stack,
  StackProps,
  Tags,
  aws_autoscaling as autoscaling,
  aws_bedrock as bedrock,
  aws_bedrockagentcore as agentcore,
  aws_cloudwatch as cw,
  aws_cloudwatch_actions as cwActions,
  aws_dynamodb as ddb,
  aws_ec2 as ec2,
  aws_events as eventsLib,
  aws_events_targets as eventTargets,
  aws_iam as iam,
  aws_kms as kms,
  aws_lambda as lambda,
  aws_logs as logs,
  aws_s3 as s3,
  aws_s3_deployment as s3deploy,
  aws_sns as sns,
  aws_sns_subscriptions as subs,
  aws_sqs as sqs,
} from "aws-cdk-lib";
import { Construct } from "constructs";

const ROOT = path.join(__dirname, "..");
const METRIC_NAMESPACE = "OpenMediaMcp";
const KNOWN_BACKENDS = ["wan2.2", "h3"];

/** Every knob is a `-c media:<key>=<value>` context value (defaults in cdk.json). */
class Config {
  constructor(private readonly scope: Construct) {}
  str(key: string, dflt = ""): string {
    const v = this.scope.node.tryGetContext(`media:${key}`);
    return v === undefined || v === null ? dflt : String(v);
  }
  num(key: string, dflt: number): number {
    const v = this.str(key, String(dflt));
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(`media:${key} must be a number, got ${v}`);
    return n;
  }
  bool(key: string, dflt: boolean): boolean {
    return ["true", "1", "yes"].includes(this.str(key, String(dflt)).toLowerCase());
  }
  list(key: string, dflt: string[] = []): string[] {
    const v = this.str(key, dflt.join(","));
    return v.split(",").map((s) => s.trim()).filter(Boolean);
  }
}

export class OpenMediaMcpStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);
    const cfg = new Config(this);
    const stackTag = "open-media-mcp:stack";
    Tags.of(this).add("project", "open-media-mcp");

    const backends = cfg.list("backends", ["wan2.2"]);
    for (const b of backends) {
      if (!KNOWN_BACKENDS.includes(b)) throw new Error(`media:backends: unknown backend ${b} (known: ${KNOWN_BACKENDS.join(", ")})`);
    }
    if (backends.includes("h3")) {
      // Not a gate, just loud: MiniMax H3 needs its own license in the US, EU, UK and Republic of Korea.
      Annotations.of(this).addWarning(
        "h3 backend enabled: MiniMax H3 Community License excludes the US, EU, UK and Republic of Korea, and companies " +
        "over USD 20M yearly revenue need MiniMax's written authorization (api@minimax.io). See THIRD_PARTY_MODELS.md.",
      );
    }
    const retain = cfg.bool("retainData", true) ? RemovalPolicy.RETAIN : RemovalPolicy.DESTROY;
    const maxInstances = cfg.num("maxInstances", 1);
    const idleMinutes = cfg.num("idleMinutes", 20);
    const presignTtl = cfg.num("presignTtlSec", 3600);

    // ---------------------------------------------------------------- data plane
    const key = cfg.bool("kms", true)
      ? new kms.Key(this, "DataKey", {
          description: "open-media-mcp media, jobs and queue encryption",
          enableKeyRotation: true,
          removalPolicy: retain,
        })
      : undefined;

    const bucket = new s3.Bucket(this, "Media", {
      encryption: key ? s3.BucketEncryption.KMS : s3.BucketEncryption.S3_MANAGED,
      encryptionKey: key,
      bucketKeyEnabled: !!key,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
      removalPolicy: retain,
      autoDeleteObjects: retain === RemovalPolicy.DESTROY,
      lifecycleRules: [
        { id: "inputs", prefix: "inputs/", expiration: Duration.days(cfg.num("inputRetentionDays", 7)) },
        { id: "outputs", prefix: "outputs/", expiration: Duration.days(cfg.num("outputRetentionDays", 30)) },
        { id: "quarantine", prefix: "quarantine/", expiration: Duration.days(cfg.num("outputRetentionDays", 30)) },
        { id: "mpu", abortIncompleteMultipartUploadAfter: Duration.days(2) },
      ],
    });

    const table = new ddb.Table(this, "Jobs", {
      partitionKey: { name: "job_id", type: ddb.AttributeType.STRING },
      billingMode: ddb.BillingMode.PAY_PER_REQUEST,
      encryption: key ? ddb.TableEncryption.CUSTOMER_MANAGED : ddb.TableEncryption.AWS_MANAGED,
      encryptionKey: key,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      timeToLiveAttribute: "expires_at",
      removalPolicy: retain,
    });
    table.addGlobalSecondaryIndex({
      indexName: "by_tenant",
      partitionKey: { name: "tenant", type: ddb.AttributeType.STRING },
      sortKey: { name: "created_at", type: ddb.AttributeType.STRING },
    });

    const dlq = new sqs.Queue(this, "JobsDlq", {
      retentionPeriod: Duration.days(14),
      encryption: key ? sqs.QueueEncryption.KMS : sqs.QueueEncryption.SQS_MANAGED,
      encryptionMasterKey: key,
      enforceSSL: true,
    });
    const queue = new sqs.Queue(this, "JobsQueue", {
      // The worker heartbeats visibility while it renders; this is only the first lease.
      visibilityTimeout: Duration.minutes(15),
      retentionPeriod: Duration.days(4),
      encryption: key ? sqs.QueueEncryption.KMS : sqs.QueueEncryption.SQS_MANAGED,
      encryptionMasterKey: key,
      dataKeyReuse: Duration.hours(1),
      enforceSSL: true,
      deadLetterQueue: { queue: dlq, maxReceiveCount: 3 },
    });

    // ---------------------------------------------------------------- guardrail
    const guardrailMode = cfg.str("guardrail", "create");
    let guardrailId = "";
    let guardrailVersion = "";
    let guardrailArn = "";
    if (guardrailMode === "create") {
      const filters = [
        ["SEXUAL", cfg.str("guardrailSexual", "HIGH")],
        ["HATE", cfg.str("guardrailHate", "HIGH")],
        ["VIOLENCE", cfg.str("guardrailViolence", "MEDIUM")],
        ["INSULTS", cfg.str("guardrailInsults", "MEDIUM")],
        ["MISCONDUCT", cfg.str("guardrailMisconduct", "MEDIUM")],
      ].filter(([, s]) => s !== "NONE");
      const g = new bedrock.CfnGuardrail(this, "PromptGuardrail", {
        name: `${Stack.of(this).stackName}-prompts`.slice(0, 50),
        description: "Screens video prompts before they reach the GPU",
        blockedInputMessaging: "This prompt was blocked by the video service content policy.",
        blockedOutputsMessaging: "Blocked by the video service content policy.",
        contentPolicyConfig: {
          filtersConfig: filters.map(([type, strength]) => ({ type, inputStrength: strength, outputStrength: "NONE" })),
        },
      });
      const v = new bedrock.CfnGuardrailVersion(this, "PromptGuardrailVersion", {
        guardrailIdentifier: g.attrGuardrailId,
        description: "deployed by open-media-mcp",
      });
      guardrailId = g.attrGuardrailId;
      guardrailVersion = v.attrVersion;
      guardrailArn = g.attrGuardrailArn;
    } else if (guardrailMode !== "none") {
      // Bring your own: "<guardrail-id>:<version>"
      [guardrailId, guardrailVersion] = guardrailMode.split(":");
      guardrailVersion = guardrailVersion || "DRAFT";
      guardrailArn = `arn:${Aws.PARTITION}:bedrock:${Aws.REGION}:${Aws.ACCOUNT_ID}:guardrail/${guardrailId}`;
    }

    // ---------------------------------------------------------------- network
    const subnetIds = cfg.list("subnetIds");
    const privateNet = cfg.bool("privateNetworking", false);
    let vpc: ec2.IVpc;
    let subnets: ec2.SubnetSelection;
    if (subnetIds.length) {
      vpc = ec2.Vpc.fromLookup(this, "Vpc", { vpcId: cfg.str("vpcId") });
      subnets = { subnets: subnetIds.map((s, i) => ec2.Subnet.fromSubnetId(this, `Subnet${i}`, s)) };
    } else {
      const v = new ec2.Vpc(this, "Vpc", {
        // More AZs = more spot pools for the GPU types.
        maxAzs: cfg.num("maxAzs", 3),
        natGateways: privateNet ? 1 : 0,
        subnetConfiguration: privateNet
          ? [
              { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
              { name: "workers", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 22 },
            ]
          : [{ name: "workers", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 22 }],
        gatewayEndpoints: {
          S3: { service: ec2.GatewayVpcEndpointAwsService.S3 },
          DynamoDB: { service: ec2.GatewayVpcEndpointAwsService.DYNAMODB },
        },
      });
      v.addFlowLog("FlowLog", { trafficType: ec2.FlowLogTrafficType.REJECT });
      vpc = v;
      subnets = { subnetType: privateNet ? ec2.SubnetType.PRIVATE_WITH_EGRESS : ec2.SubnetType.PUBLIC };
    }
    const workerSg = new ec2.SecurityGroup(this, "WorkerSg", {
      vpc,
      description: "open-media-mcp GPU worker, no inbound rules (pull-based via SQS)",
      allowAllOutbound: true,
    });

    // ---------------------------------------------------------------- worker
    const workerLogs = new logs.LogGroup(this, "WorkerLogs", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: RemovalPolicy.DESTROY,
    });
    const workerRole = new iam.Role(this, "WorkerRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      description: "open-media-mcp GPU worker",
      managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore")],
    });
    bucket.grantReadWrite(workerRole);
    table.grantReadWriteData(workerRole);
    queue.grantConsumeMessages(workerRole);
    workerLogs.grantWrite(workerRole);
    const modelCacheBucket = cfg.str("modelCacheBucket");
    if (modelCacheBucket) {
      // Optional pre-seeded weights in another bucket (read only), checked before Hugging Face.
      workerRole.addToPolicy(new iam.PolicyStatement({
        actions: ["s3:GetObject", "s3:ListBucket"],
        resources: [`arn:${Aws.PARTITION}:s3:::${modelCacheBucket}`, `arn:${Aws.PARTITION}:s3:::${modelCacheBucket}/models/*`],
      }));
    }
    workerRole.addToPolicy(new iam.PolicyStatement({ actions: ["rekognition:DetectModerationLabels"], resources: ["*"] }));
    workerRole.addToPolicy(new iam.PolicyStatement({
      actions: ["events:PutEvents"],
      resources: [`arn:${Aws.PARTITION}:events:${Aws.REGION}:${Aws.ACCOUNT_ID}:event-bus/default`],
    }));
    workerRole.addToPolicy(new iam.PolicyStatement({
      actions: ["cloudwatch:PutMetricData"],
      resources: ["*"],
      conditions: { StringEquals: { "cloudwatch:namespace": METRIC_NAMESPACE } },
    }));
    workerRole.addToPolicy(new iam.PolicyStatement({
      // A worker only ever removes itself; scoped by tag to avoid an ASG <-> role dependency cycle.
      actions: ["autoscaling:TerminateInstanceInAutoScalingGroup", "autoscaling:SetDesiredCapacity"],
      resources: ["*"],
      conditions: { StringEquals: { [`autoscaling:ResourceTag/${stackTag}`]: this.stackName } },
    }));
    workerRole.addToPolicy(new iam.PolicyStatement({ actions: ["autoscaling:DescribeAutoScalingInstances"], resources: ["*"] }));
    const hfSecret = cfg.str("hfTokenSecretName");
    if (hfSecret) {
      workerRole.addToPolicy(new iam.PolicyStatement({
        actions: ["secretsmanager:GetSecretValue"],
        resources: [`arn:${Aws.PARTITION}:secretsmanager:${Aws.REGION}:${Aws.ACCOUNT_ID}:secret:${hfSecret}*`],
      }));
    }

    new s3deploy.BucketDeployment(this, "WorkerCode", {
      sources: [s3deploy.Source.asset(path.join(ROOT, "worker"), { exclude: ["__pycache__", "*.pyc"] })],
      destinationBucket: bucket,
      destinationKeyPrefix: "bootstrap/",
      memoryLimit: 512,
    });

    const workerEnv: Record<string, string> = {
      MEDIA_STACK: this.stackName,
      MEDIA_BACKENDS: backends.join(","),
      MEDIA_BUCKET: bucket.bucketName,
      MEDIA_MODEL_BUCKETS: [bucket.bucketName, modelCacheBucket].filter(Boolean).join(","),
      MEDIA_TABLE: table.tableName,
      MEDIA_QUEUE_URL: queue.queueUrl,
      MEDIA_LOG_GROUP: workerLogs.logGroupName,
      MEDIA_IDLE_MINUTES: String(idleMinutes),
      MEDIA_MODERATION: String(cfg.bool("outputModeration", true)),
      MEDIA_MODERATION_MIN_CONFIDENCE: String(cfg.num("moderationMinConfidence", 80)),
      MEDIA_BLOCKED_LABELS: cfg.list("blockedLabels", ["Explicit", "Violence", "Visually Disturbing", "Hate Symbols"]).join(","),
      MEDIA_HF_TOKEN_SECRET: hfSecret,
      MEDIA_METRIC_NAMESPACE: METRIC_NAMESPACE,
      MEDIA_JOB_RETENTION_DAYS: String(cfg.num("jobRetentionDays", 90)),
      MEDIA_COMFYUI_REF: cfg.str("comfyuiRef", "master"),
      AWS_DEFAULT_REGION: Aws.REGION,
    };
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      "set -euo pipefail",
      "mkdir -p /opt/media-bootstrap",
      // Double-quoted values: valid for both bash `source` and systemd EnvironmentFile (labels contain spaces).
      `cat > /etc/media-worker.env <<'EOF'\n${Object.entries(workerEnv).map(([k, v]) => `${k}="${v}"`).join("\n")}\nEOF`,
      `aws s3 cp --recursive s3://${bucket.bucketName}/bootstrap/ /opt/media-bootstrap/ --only-show-errors`,
      "chmod +x /opt/media-bootstrap/*.sh",
      "nohup /opt/media-bootstrap/setup.sh >/var/log/media-setup.log 2>&1 &",
    );

    const launchTemplate = new ec2.LaunchTemplate(this, "WorkerTemplate", {
      machineImage: ec2.MachineImage.fromSsmParameter(
        cfg.str("amiParameter", "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"),
      ),
      role: workerRole,
      securityGroup: workerSg,
      userData,
      requireImdsv2: true,
      instanceMetadataTags: true,
      associatePublicIpAddress: privateNet || subnetIds.length ? undefined : true,
      blockDevices: [{
        deviceName: "/dev/sda1",
        volume: ec2.BlockDeviceVolume.ebs(cfg.num("rootVolumeGb", 200), {
          volumeType: ec2.EbsDeviceVolumeType.GP3, iops: 6000, throughput: 500, encrypted: true, deleteOnTermination: true,
        }),
      }],
    });

    const instanceTypes = cfg.list("instanceTypes", ["g7e.4xlarge", "g7e.2xlarge", "g6e.4xlarge"]);
    const asg = new autoscaling.AutoScalingGroup(this, "Workers", {
      vpc,
      vpcSubnets: subnets,
      minCapacity: 0,
      maxCapacity: maxInstances,
      // desiredCapacity left unset on purpose: the tools Lambda and the idle workers own it.
      mixedInstancesPolicy: {
        launchTemplate,
        launchTemplateOverrides: instanceTypes.map((t) => ({ instanceType: new ec2.InstanceType(t) })),
        instancesDistribution: {
          onDemandBaseCapacity: 0,
          onDemandPercentageAboveBaseCapacity: cfg.bool("onDemand", false) ? 100 : 0,
          onDemandAllocationStrategy: autoscaling.OnDemandAllocationStrategy.PRIORITIZED,
          spotAllocationStrategy: autoscaling.SpotAllocationStrategy.CAPACITY_OPTIMIZED_PRIORITIZED,
        },
      },
      capacityRebalance: false,
    });
    Tags.of(asg).add(stackTag, this.stackName, { applyToLaunchedInstances: false });
    Tags.of(asg).add("Name", `${this.stackName}-worker`, { applyToLaunchedInstances: true });

    // ---------------------------------------------------------------- MCP tools
    const toolsFn = new lambda.Function(this, "ToolsFn", {
      runtime: lambda.Runtime.PYTHON_3_13,
      architecture: lambda.Architecture.ARM_64,
      handler: "handler.lambda_handler",
      code: lambda.Code.fromAsset(path.join(ROOT, "lambda/tools")),
      timeout: Duration.seconds(30),
      memorySize: 256,
      description: "open-media-mcp MCP tools (AgentCore Gateway Lambda target)",
      logGroup: new logs.LogGroup(this, "ToolsLogs", { retention: logs.RetentionDays.THREE_MONTHS, removalPolicy: RemovalPolicy.DESTROY }),
      environment: {
        MEDIA_BACKENDS: backends.join(","),
        MEDIA_TABLE: table.tableName,
        MEDIA_QUEUE_URL: queue.queueUrl,
        MEDIA_BUCKET: bucket.bucketName,
        MEDIA_ASG: asg.autoScalingGroupName,
        MEDIA_MAX_INSTANCES: String(maxInstances),
        MEDIA_PRESIGN_TTL: String(presignTtl),
        MEDIA_GUARDRAIL_ID: guardrailId,
        MEDIA_GUARDRAIL_VERSION: guardrailVersion,
        MEDIA_DAILY_JOB_LIMIT: String(cfg.num("dailyJobLimitPerCaller", 100)),
        MEDIA_MAX_QUEUED: String(cfg.num("maxQueuedJobs", 50)),
        MEDIA_JOB_RETENTION_DAYS: String(cfg.num("jobRetentionDays", 90)),
        MEDIA_MODERATION: String(cfg.bool("outputModeration", true)),
        MEDIA_IDLE_MINUTES: String(idleMinutes),
        MEDIA_INSTANCE_TYPES: instanceTypes.join(","),
      },
    });
    table.grantReadWriteData(toolsFn);
    queue.grantSendMessages(toolsFn);
    queue.grant(toolsFn, "sqs:GetQueueAttributes");
    bucket.grantPut(toolsFn, "inputs/*");
    bucket.grantRead(toolsFn, "inputs/*");
    bucket.grantRead(toolsFn, "outputs/*");
    bucket.grantRead(toolsFn, "status/*");
    toolsFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["autoscaling:SetDesiredCapacity", "autoscaling:UpdateAutoScalingGroup"],
      resources: ["*"],
      conditions: { StringEquals: { [`autoscaling:ResourceTag/${stackTag}`]: this.stackName } },
    }));
    toolsFn.addToRolePolicy(new iam.PolicyStatement({
      actions: ["autoscaling:DescribeAutoScalingGroups", "ec2:DescribeLaunchTemplateVersions", "ec2:DescribeInstances"],
      resources: ["*"],
    }));
    // UpdateAutoScalingGroup re-validates the launch template, which carries the worker role.
    toolsFn.addToRolePolicy(new iam.PolicyStatement({ actions: ["iam:PassRole"], resources: [workerRole.roleArn] }));
    if (guardrailArn) {
      toolsFn.addToRolePolicy(new iam.PolicyStatement({ actions: ["bedrock:ApplyGuardrail"], resources: [guardrailArn] }));
    }

    // Safety net: queued jobs always get a GPU, even if a worker scaled itself in during a submit.
    new eventsLib.Rule(this, "Reconcile", {
      schedule: eventsLib.Schedule.rate(Duration.minutes(2)),
      targets: [new eventTargets.LambdaFunction(toolsFn, { event: eventsLib.RuleTargetInput.fromObject({ reconcile: true }) })],
    });

    const interceptorFn = new lambda.Function(this, "IdentityInterceptorFn", {
      runtime: lambda.Runtime.PYTHON_3_13,
      architecture: lambda.Architecture.ARM_64,
      handler: "handler.lambda_handler",
      code: lambda.Code.fromAsset(path.join(ROOT, "lambda/interceptor")),
      timeout: Duration.seconds(10),
      memorySize: 128,
      description: "open-media-mcp gateway REQUEST interceptor: stamps the verified caller onto every tool call",
      logGroup: new logs.LogGroup(this, "InterceptorLogs", { retention: logs.RetentionDays.THREE_MONTHS, removalPolicy: RemovalPolicy.DESTROY }),
      environment: { MEDIA_TENANT_CLAIM: cfg.str("tenantClaim", "sub") },
    });
    interceptorFn.addToRolePolicy(new iam.PolicyStatement({ actions: ["sts:GetAccessKeyInfo"], resources: ["*"] }));

    // ---------------------------------------------------------------- AgentCore Gateway
    const gatewayRole = new iam.Role(this, "GatewayRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com", {
        conditions: { StringEquals: { "aws:SourceAccount": Aws.ACCOUNT_ID } },
      }),
      description: "AgentCore Gateway role for open-media-mcp (invokes the tool and interceptor Lambdas)",
    });
    toolsFn.grantInvoke(gatewayRole);
    interceptorFn.grantInvoke(gatewayRole);

    const auth = cfg.str("auth", "iam");
    let authorizerConfiguration: agentcore.CfnGateway.AuthorizerConfigurationProperty | undefined;
    if (auth === "jwt") {
      const discoveryUrl = cfg.str("jwtDiscoveryUrl");
      if (!discoveryUrl) throw new Error("media:auth=jwt needs media:jwtDiscoveryUrl");
      const clients = cfg.list("jwtAllowedClients");
      const audience = cfg.list("jwtAllowedAudience");
      authorizerConfiguration = {
        customJwtAuthorizer: {
          discoveryUrl,
          allowedClients: clients.length ? clients : undefined,
          allowedAudience: audience.length ? audience : undefined,
        },
      };
    } else if (auth !== "iam") {
      throw new Error("media:auth must be iam or jwt (unauthenticated gateways are not supported)");
    }

    const gateway = new agentcore.CfnGateway(this, "Gateway", {
      name: cfg.str("gatewayName", this.stackName.toLowerCase().replace(/[^0-9a-z-]/g, "-")).slice(0, 48),
      description: "Open-weight video generation tools (Wan 2.2, MiniMax-H3)",
      roleArn: gatewayRole.roleArn,
      protocolType: "MCP",
      authorizerType: auth === "jwt" ? "CUSTOM_JWT" : "AWS_IAM",
      authorizerConfiguration,
      protocolConfiguration: {
        mcp: {
          instructions:
            "Video generation is asynchronous. Call get_capabilities first. For each input file, call create_upload " +
            "and PUT the bytes to the returned URL. Then call generate_video with the returned keys, and poll get_job " +
            "every 30-60 s until it reaches a final status. A cold GPU takes about 5 minutes to start; gpu_capacity " +
            "warm starts one ahead of time.",
        },
      },
      interceptorConfigurations: [{
        interceptor: { lambda: { arn: interceptorFn.functionArn } },
        interceptionPoints: ["REQUEST"],
        inputConfiguration: { passRequestHeaders: true },
      }],
      exceptionLevel: cfg.bool("gatewayDebug", false) ? "DEBUG" : undefined,
    });
    gateway.node.addDependency(gatewayRole);

    const tools = JSON.parse(fs.readFileSync(path.join(ROOT, "tools.json"), "utf8"));
    const target = new agentcore.CfnGatewayTarget(this, "ToolsTarget", {
      gatewayIdentifier: gateway.attrGatewayIdentifier,
      name: cfg.str("toolPrefix", "media"),
      description: "open-media-mcp generation tools",
      credentialProviderConfigurations: [{ credentialProviderType: "GATEWAY_IAM_ROLE" }],
      targetConfiguration: {
        mcp: { lambda: { lambdaArn: toolsFn.functionArn, toolSchema: { inlinePayload: tools.map(toCfnTool) } } },
      },
    });
    target.node.addDependency(gatewayRole);

    // ---------------------------------------------------------------- observability
    const topic = new sns.Topic(this, "Alarms", { masterKey: key });
    // CloudWatch alarms publish to the encrypted topic.
    key?.addToResourcePolicy(new iam.PolicyStatement({
      principals: [new iam.ServicePrincipal("cloudwatch.amazonaws.com")],
      actions: ["kms:Decrypt", "kms:GenerateDataKey*"],
      resources: ["*"],
      conditions: { StringEquals: { "aws:SourceAccount": Aws.ACCOUNT_ID } },
    }));
    const alarmEmail = cfg.str("alarmEmail");
    if (alarmEmail) topic.addSubscription(new subs.EmailSubscription(alarmEmail));
    const m = (name: string, stat = "Sum") =>
      new cw.Metric({ namespace: METRIC_NAMESPACE, metricName: name, dimensionsMap: { Stack: this.stackName }, statistic: stat, period: Duration.minutes(5) });

    const dlqAlarm = dlq.metricApproximateNumberOfMessagesVisible({ period: Duration.minutes(5) })
      .createAlarm(this, "DeadLetters", { threshold: 0, comparisonOperator: cw.ComparisonOperator.GREATER_THAN_THRESHOLD, evaluationPeriods: 1, alarmDescription: "A job failed 3 times and was dead-lettered" });
    const failAlarm = m("JobFailed").with({ period: Duration.hours(1) })
      .createAlarm(this, "JobFailures", { threshold: 3, evaluationPeriods: 1, treatMissingData: cw.TreatMissingData.NOT_BREACHING, alarmDescription: "3 or more failed jobs in an hour" });
    const stuckAlarm = queue.metricApproximateAgeOfOldestMessage({ period: Duration.minutes(5) })
      .createAlarm(this, "QueueStuck", { threshold: 3600, evaluationPeriods: 2, alarmDescription: "Oldest queued job is over an hour old (no GPU capacity?)" });
    for (const a of [dlqAlarm, failAlarm, stuckAlarm]) a.addAlarmAction(new cwActions.SnsAction(topic));

    const dash = new cw.Dashboard(this, "Dashboard", { dashboardName: `${this.stackName}-ops`.slice(0, 255) });
    dash.addWidgets(
      new cw.GraphWidget({ title: "Queue", left: [queue.metricApproximateNumberOfMessagesVisible(), queue.metricApproximateNumberOfMessagesNotVisible(), dlq.metricApproximateNumberOfMessagesVisible({ label: "dead letters" })] }),
      new cw.GraphWidget({ title: "Jobs", left: [m("JobSucceeded"), m("JobFailed"), m("JobBlocked"), m("JobCancelled")] }),
      new cw.GraphWidget({ title: "Render and wait (s)", left: [m("RenderSeconds", "p50"), m("RenderSeconds", "p90"), m("QueueWaitSeconds", "p90")] }),
      new cw.GraphWidget({ title: "GPU workers", left: [new cw.Metric({ namespace: "AWS/AutoScaling", metricName: "GroupInServiceInstances", dimensionsMap: { AutoScalingGroupName: asg.autoScalingGroupName }, statistic: "Maximum", period: Duration.minutes(5) })] }),
      new cw.GraphWidget({ title: "Tool calls", left: [toolsFn.metricInvocations(), toolsFn.metricErrors()] }),
    );

    // ---------------------------------------------------------------- outputs
    new CfnOutput(this, "GatewayUrl", { value: gateway.attrGatewayUrl, description: "MCP endpoint (SigV4 or JWT)" });
    new CfnOutput(this, "GatewayId", { value: gateway.attrGatewayIdentifier });
    new CfnOutput(this, "GatewayArn", { value: gateway.attrGatewayArn });
    new CfnOutput(this, "BucketName", { value: bucket.bucketName });
    new CfnOutput(this, "JobsTable", { value: table.tableName });
    new CfnOutput(this, "QueueUrl", { value: queue.queueUrl });
    new CfnOutput(this, "WorkerGroup", { value: asg.autoScalingGroupName });
    new CfnOutput(this, "WorkerLogGroup", { value: workerLogs.logGroupName });
    new CfnOutput(this, "DashboardUrl", { value: `https://${Aws.REGION}.console.aws.amazon.com/cloudwatch/home?region=${Aws.REGION}#dashboards:name=${dash.dashboardName}` });
  }
}

/** tools.json (MCP casing) -> the CloudFormation ToolDefinition shape (no enum/default support there). */
function toCfnTool(t: any): agentcore.CfnGatewayTarget.ToolDefinitionProperty {
  return { name: t.name, description: t.description, inputSchema: toCfnSchema(t.inputSchema) };
}
function toCfnSchema(s: any): agentcore.CfnGatewayTarget.SchemaDefinitionProperty {
  const out: any = { type: s.type };
  if (s.description) out.description = s.description;
  if (s.properties) out.properties = Object.fromEntries(Object.entries(s.properties).map(([k, v]) => [k, toCfnSchema(v)]));
  if (s.required) out.required = s.required;
  if (s.items) out.items = toCfnSchema(s.items);
  return out;
}
