package br.com.microservices.orchestrated.orchestratorservice.core.saga;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

import static br.com.microservices.orchestrated.orchestratorservice.core.enums.EEventSource.*;
import static br.com.microservices.orchestrated.orchestratorservice.core.enums.ESagaStatus.*;
import static br.com.microservices.orchestrated.orchestratorservice.core.enums.ETopics.*;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class SagaHandler {

    public static final int EVENT_SOURCE_INDEX = 0;
    public static final int SAGA_STATUS_INDEX = 1;
    public static final int TOPIC_INDEX = 2;

    public static final Object[][] SAGA_HANDLER = {
            {ORCHESTRATOR, SUCCESS, DOCUMENT_BOT_SUCCESS},
            {ORCHESTRATOR, FAIL, FINISH_FAIL},

            {DOCUMENT_BOT_SERVICE, ROLLBACK_PENDING, DOCUMENT_BOT_FAIL},
            {DOCUMENT_BOT_SERVICE, FAIL, FINISH_FAIL},
            {DOCUMENT_BOT_SERVICE, SUCCESS, FINISH_SUCCESS}
    };
}
