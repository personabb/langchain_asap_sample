import json
import re
import os
import sys
import operator
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_mcp_adapters.client import MultiServerMCPClient

_ = load_dotenv(find_dotenv())
google_api_key = os.getenv("GOOGLE_APIKEY")

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def create_graph(state: GraphState, tools, model_chain):
    def should_continue(state: state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END


    def call_model(state: state):
        messages = state["messages"]
        response = model_chain.invoke(messages)
        return {"messages": [response]}


    tool_node = ToolNode(tools)
    
    workflow = StateGraph(state)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app



async def main(graph_config = {"configurable": {"thread_id": "12345"}}):
    # モデルの定義。APIキーは環境変数から取得
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key,
        temperature=0.001,
        )

    with open("mcp_config.json", "r") as f:
        mcp_config = json.load(f)

    # messageを作成する
    message = [
        SystemMessage(content= """
あなたは役にたつAIアシスタントです。日本語で回答し、考えた過程を結論より前に出力してください。
あなたは、「PlayWrite」というブラウザを操作するtoolを利用することができます。適切に利用してユーザからの質問に回答してください。
ツールを利用する場合は、必ずツールから得られた情報のみを利用して回答してください。

まず、ユーザの質問からツールをどういう意図で何回利用しないといけないのかを判断し、必要なら複数回toolを利用して情報収集をしたのち、すべての情報が取得できたら、その情報を元に返答してください。

なお、サイトのアクセスでエラーが出た場合は、もう一度再施行してください。ネットワーク関連のエラーの場合があります。
    """),
        MessagesPlaceholder("messages"),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    async with MultiServerMCPClient(mcp_config["mcpServers"]) as mcp_client:
        tools = mcp_client.get_tools()

        model_with_tools = prompt | model.bind_tools(tools)

        graph = create_graph(
            GraphState,
            tools,
            model_with_tools
        )


        while True:
            query = input("入力してください: ") 

            if query.lower() in ["exit", "quit"]:
                print("終了します。")
                break

            input_query = [HumanMessage(
                    [
                        {
                            "type": "text",
                            "text": f"{query}"
                        },
                    ]
                )]

            response = await graph.ainvoke({"messages":input_query}, graph_config)

            #デバック用
            print("response: ", response)

            # 最終的な回答
            print("=================================")
            print(response["messages"][-1].content)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

