{{/*
Expand the name of the chart.
*/}}
{{- define "microservice-platform.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "microservice-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "microservice-platform.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "microservice-platform.labels" -}}
helm.sh/chart: {{ include "microservice-platform.chart" . }}
{{ include "microservice-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "microservice-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ include "microservice-platform.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name for the gateway
*/}}
{{- define "microservice-platform.gateway.fullname" -}}
{{- printf "%s-gateway" (include "microservice-platform.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the name for the user service
*/}}
{{- define "microservice-platform.userService.fullname" -}}
{{- printf "%s-user-service" (include "microservice-platform.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the name for the auth service
*/}}
{{- define "microservice-platform.authService.fullname" -}}
{{- printf "%s-auth-service" (include "microservice-platform.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the name for the billing service
*/}}
{{- define "microservice-platform.billingService.fullname" -}}
{{- printf "%s-billing-service" (include "microservice-platform.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the name for the notification service
*/}}
{{- define "microservice-platform.notificationService.fullname" -}}
{{- printf "%s-notification-service" (include "microservice-platform.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Return the appropriate apiVersion for HPA
*/}}
{{- define "microservice-platform.hpa.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "autoscaling/v2" }}
{{- print "autoscaling/v2" }}
{{- else }}
{{- print "autoscaling/v2beta2" }}
{{- end }}
{{- end }}
