package br.com.microservices.orchestrated.orchestratorservice.core.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum ETopics {

    START_SAGA("start-saga"),
    BASE_ORCHESTRATOR("orchestrator"),
    FINISH_SUCCESS("finish-success"),
    FINISH_FAIL("finish-fail"),
    DOCUMENT_BOT_SUCCESS("document-bot-success"),
    DOCUMENT_BOT_FAIL("document-bot-fail"),
    NOTIFY_ENDING("notify-ending");

    private final String topic;
}
