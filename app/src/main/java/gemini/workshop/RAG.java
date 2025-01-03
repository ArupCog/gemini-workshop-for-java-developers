/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package gemini.workshop;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.model.vertexai.VertexAiGeminiChatModel;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.List;

public class RAG {

    interface LlmExpert {
        String ask(String question);
    }

    public static void main(String[] args) throws IOException, URISyntaxException {

        URL url = new URI("https://github.com/ArupCog/gemini-workshop-for-java-developers/raw/main/vantage-test-cases.pdf").toURL();
        ApachePdfBoxDocumentParser pdfParser = new ApachePdfBoxDocumentParser();
        Document document = pdfParser.parse(url.openStream());
        //Document document = pdfParser.parse(new FileInputStream("/tmp/attention-is-all-you-need.pdf"));

        VertexAiEmbeddingModel embeddingModel = VertexAiEmbeddingModel.builder()
            .endpoint(System.getenv("LOCATION") + "-aiplatform.googleapis.com:443")
            .project(System.getenv("PROJECT_ID"))
            .location(System.getenv("LOCATION"))
            .publisher("google")
            .modelName("textembedding-gecko@003")
            .maxRetries(3)
            .build();

        InMemoryEmbeddingStore<TextSegment> embeddingStore =
            new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor storeIngestor = EmbeddingStoreIngestor.builder()
            .documentSplitter(DocumentSplitters.recursive(500, 100))
            .embeddingModel(embeddingModel)
            .embeddingStore(embeddingStore)
            .build();
        System.out.println("Chunking and embedding PDF...");
        storeIngestor.ingest(document);

        ChatLanguageModel model = VertexAiGeminiChatModel.builder()
                .project(System.getenv("PROJECT_ID"))
                .location(System.getenv("LOCATION"))
                .modelName("gemini-1.5-flash-001")
                .maxOutputTokens(1000)
                .build();

        EmbeddingStoreContentRetriever retriever =
            new EmbeddingStoreContentRetriever(embeddingStore, embeddingModel);

        LlmExpert expert = AiServices.builder(LlmExpert.class)
            .chatLanguageModel(model)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .contentRetriever(retriever)
            /*
            .retrievalAugmentor(DefaultRetrievalAugmentor.builder()
                .contentInjector(DefaultContentInjector.builder()
                    .promptTemplate(PromptTemplate.from("""
                        You are an expert in large language models,\s
                        you excel at explaining simply and clearly questions about LLMs.

                        Here is the question: {{userMessage}}

                        Answer using the following information:
                        {{contents}}
                        """))
                    .build())
                .contentRetriever(retriever)
                .build())
             */
            .build();

        System.out.println("Ready!\n");
        List.of(
            "Leverage the historical test cases as a reference point for understanding existing functionality and potential test scenarios, which contains test scenarios for creating prospects."+
            "Consider the below points while writting the test cases."+
            "1)Write detailed test cases for update and delete prospects in a structured readable format.Consider all positive and negative scenarios while creating the test cases"+
            "2)Include both positive and negative test cases."+
            "3)For each test case, provide a unique ID and a clear description."+
            "4)Detail each test step, including expected results.Make sure each test steps are in separate lines while generating the test cases"+
            "5)Incorporate iCABS steps where applicable, providing clear and concise instructions."
        ).forEach(query ->
            System.out.printf("%n=== %s === %n%n %s %n%n", query, expert.ask(query)));
    }
}
