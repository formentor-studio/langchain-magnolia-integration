import { MagnoliaContentsLoader } from "./document_loaders/web/magnolia.js"; // NOTE: Requires local installation of https://github.com/joaquin-alfaro/langchainjs/tree/feature/magnolia-loader locally 
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { VectorStore } from "@langchain/core/vectorstores";

function createVectorStore(): VectorStore {
    const embeddings = new OpenAIEmbeddings({
        apiKey: process.env.OPENAI_API_KEY, // In Node.js defaults to process.env.OPENAI_API_KEY
        model: "text-embedding-3-small",
    })

    return new MemoryVectorStore(embeddings) as unknown as VectorStore
}

async function loadContentsAndQuestion() {
    /**
     * 1. Load contents from Magnolia
     */
    const loader = new MagnoliaContentsLoader({
        collection: "tours",
        baseUrl: "http://localhost:8080/.rest/delivery/tours/v1",
        contentProperty: "body"
    });
    const docs = await loader.load();

    /**
     * 2. Split documents in chunks before embedding
     */
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 180,
        chunkOverlap: 14,
    })
    const chunks = await splitter.splitDocuments(docs);

    /**
     * 3. Embed documents and store in Vector store
     */
    const vectorStore = createVectorStore()
    await vectorStore.addDocuments(chunks)

    /**
     * 4. Search contents by similarity
     */
    const question = "Find tours for cycling"
    const similarDocs = await vectorStore.similaritySearch(question, 3, (doc: Document | any)  => doc.metadata.collection == "tours")
    similarDocs.map((doc) => {
        console.log(doc)
        console.log('\n')
    })
}

loadContentsAndQuestion()
